"""QACap model: BLIP caption decoder + ViT image encoder + BERT question encoder.

This module follows the architecture described in Sections 3.3.1--3.3.3 and
Figure 5 of "Separation of Powers" (CVPR 2025):

  * pretrained ViT image features F_v;
  * BERT question features F_q;
  * QAFD: MHSA(F_q,F_q,F_q), then MHCA(question,image,image), followed by
    the token-wise and sentence-wise visual reweighting in Eqs. (3)--(8);
  * BLIP text decoder over PET-prefixed caption tokens, cross-attending to
    concatenate([F_tilde_q, F_tilde_v]);
  * QARCL attention maps exposed to the training/loss code.

The paper's main text does not state the numerical alpha value or the exact
number of repeated QAFD blocks (shown as N x in Figure 5). They are therefore
required constructor arguments rather than silently chosen defaults.

Expected training batch fields are compatible with data_updated.py:
    pixel_values, questions, decoder_input_ids, decoder_attention_mask

The caption targets and PET-prefix masking are prepared by the Dataset; the
caller computes L_cap and lambda_JS * L_JS using losses.py.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


PET_PROMPT_TEMPLATE = (
    "Please describe this image according to the following question: {question}"
)


class _QAFDBlock(nn.Module):
    """One QAFD block: question MHSA, question-to-image MHCA and FFN.

    Figure 5 explicitly places a Feed Forward block after QAFD cross-attention.
    Its internal width/activation/residual details are not specified in the
    paper's main text, so the implementation settings are exposed in config.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        ffn_ratio: float = 4.0,
        ffn_activation: str = "relu",
        ffn_residual: bool = False,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be positive and divide dim ({dim})"
            )
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be positive, got {ffn_ratio}")
        activation_name = str(ffn_activation).lower()
        activations = {"relu": nn.ReLU, "gelu": nn.GELU}
        if activation_name not in activations:
            raise ValueError(
                f"ffn_activation must be one of {tuple(activations)}, got {ffn_activation!r}"
            )
        self.ffn_residual = bool(ffn_residual)
        intermediate_dim = max(1, int(round(dim * float(ffn_ratio))))

        self.question_mhsa = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.question_to_image_mhca = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            activations[activation_name](),
            nn.Linear(intermediate_dim, dim),
        )

    def forward(
        self,
        question_features: torch.Tensor,
        image_features: torch.Tensor,
        question_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Eq. (1): F_hat_q = MHSA(F_q, F_q, F_q).
        q_hat, _ = self.question_mhsa(
            query=question_features,
            key=question_features,
            value=question_features,
            key_padding_mask=question_padding_mask,
            need_weights=False,
        )
        if question_padding_mask is not None:
            q_hat = q_hat.masked_fill(question_padding_mask.unsqueeze(-1), 0.0)

        # Eq. (2): F_tilde_q = MHCA(F_hat_q, F_v, F_v).
        q_cross, attention = self.question_to_image_mhca(
            query=q_hat,
            key=image_features,
            value=image_features,
            need_weights=True,
            average_attn_weights=True,
        )
        # Figure 5 shows a Feed Forward block after the QAFD cross-attention.
        # The main text does not specify its expansion ratio, activation, or
        # residual convention; those choices are surfaced as config settings.
        q_tilde = self.feed_forward(q_cross)
        if self.ffn_residual:
            q_tilde = q_cross + q_tilde
        if question_padding_mask is not None:
            q_tilde = q_tilde.masked_fill(question_padding_mask.unsqueeze(-1), 0.0)
            attention = attention.masked_fill(question_padding_mask.unsqueeze(-1), 0.0)
        # attention: [batch, question_tokens, image_tokens], i.e. A in R^(Lq x Lv).
        return q_hat, q_tilde, attention


class QAFD(nn.Module):
    """Question-Aware Feature Distilling module (paper Eqs. 1--8)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        alpha: float,
        num_layers: int,
        ffn_ratio: float = 4.0,
        ffn_activation: str = "relu",
        ffn_residual: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.alpha = float(alpha)
        self.ffn_ratio = float(ffn_ratio)
        self.ffn_activation = str(ffn_activation).lower()
        self.ffn_residual = bool(ffn_residual)
        self.layers = nn.ModuleList(
            [
                _QAFDBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_ratio=self.ffn_ratio,
                    ffn_activation=self.ffn_activation,
                    ffn_residual=self.ffn_residual,
                )
                for _ in range(num_layers)
            ]
        )
        self.last_attention: Optional[torch.Tensor] = None
        self.last_question_features: Optional[torch.Tensor] = None
        self.last_distilled_image_features: Optional[torch.Tensor] = None

    def forward(
        self,
        question_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        question_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if question_tokens.ndim != 3 or image_tokens.ndim != 3:
            raise ValueError(
                "question_tokens and image_tokens must be rank-3 tensors "
                "[batch, sequence, hidden]"
            )
        if question_tokens.shape[0] != image_tokens.shape[0]:
            raise ValueError("Question and image batch sizes do not match")
        if question_tokens.shape[-1] != image_tokens.shape[-1]:
            raise ValueError(
                "QAFD requires matching question/image feature dimensions; got "
                f"{question_tokens.shape[-1]} and {image_tokens.shape[-1]}"
            )
        if question_padding_mask is not None:
            if question_padding_mask.shape != question_tokens.shape[:2]:
                raise ValueError(
                    "question_padding_mask must have shape [batch, question_tokens]"
                )
            question_padding_mask = question_padding_mask.to(
                device=question_tokens.device, dtype=torch.bool
            )
            if question_padding_mask.all(dim=1).any():
                raise ValueError("Each question must contain at least one non-padding token")

        q = question_tokens
        q_hat = None
        attention = None
        for layer in self.layers:
            q_hat, q, attention = layer(q, image_tokens, question_padding_mask)

        if q_hat is None or attention is None:
            raise RuntimeError("QAFD did not produce question features and attention")

        # Eq. (3): column-wise maximum over valid question tokens.
        if question_padding_mask is not None:
            attention_for_max = attention.masked_fill(
                question_padding_mask.unsqueeze(-1), float("-inf")
            )
            q_for_pool = q_hat.masked_fill(
                question_padding_mask.unsqueeze(-1), float("-inf")
            )
        else:
            attention_for_max = attention
            q_for_pool = q_hat
        token_similarity = attention_for_max.amax(dim=1)  # [B, L_v]

        # Eq. (4): F_hat_qs = max_pool(F_hat_q).
        question_sentence = q_for_pool.amax(dim=1)  # [B, D]

        # Eq. (5): S_w = cosine(F_v, F_hat_qs); retain signed cosine values.
        sentence_similarity = F.cosine_similarity(
            image_tokens, question_sentence.unsqueeze(1), dim=-1
        )  # [B, L_v]

        # Eq. (6): S_o = (S_w + S_q) / 2.
        region_weights = 0.5 * (sentence_similarity + token_similarity)

        # Eqs. (7)--(8): F_hat_v = F_v * S_o; F_tilde_v = alpha*F_v + (1-alpha)*F_hat_v.
        enhanced_image = image_tokens * region_weights.unsqueeze(-1)
        distilled_image = (
            self.alpha * image_tokens + (1.0 - self.alpha) * enhanced_image
        )

        self.last_attention = attention
        self.last_question_features = q
        self.last_distilled_image_features = distilled_image
        return distilled_image, q, attention


class QACap(nn.Module):
    """Complete QACap captioner with paper-defined QAFD and QARCL maps.

    Args:
        blip_name: pretrained BLIP checkpoint (paper says QACap is fine-tuned
            from BLIP; the base checkpoint must match the experiment config).
        bert_name: pretrained BERT checkpoint for question encoding.
        alpha: QAFD mean-weighted factor from Eq. (8). Required because its
            numerical value is not stated in the main paper text.
        num_heads: head count for QAFD MHSA/MHCA. Required because the main paper
            does not give this numeric value.
        qafd_layers: repeated QAFD block count shown as N x in Figure 5. Required
            because the main paper does not give N numerically.
        max_question_length: optional explicit token limit. None uses the
            tokenizer's configured model limit.

    ``forward`` returns ``(logits, qafd_attention)`` for compatibility with the
    training loop. The final decoder cross-attention map is available as
    ``last_decoder_attention``. Both maps are [B, query_tokens, image_tokens].
    """

    def __init__(
        self,
        blip_name: str = "Salesforce/blip-image-captioning-base",
        bert_name: str = "bert-base-uncased",
        *,
        alpha: float,
        num_heads: int,
        qafd_layers: int,
        qafd_ffn_ratio: float = 4.0,
        qafd_ffn_activation: str = "relu",
        qafd_ffn_residual: bool = False,
        max_question_length: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Lazy import keeps QAFD unit tests usable without Transformers installed.
        try:
            from transformers import (
                BertModel,
                BertTokenizer,
                BlipForConditionalGeneration,
                BlipProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "QACap requires Hugging Face Transformers. Install the pinned "
                "project requirements before constructing QACap."
            ) from exc

        if max_question_length is not None and max_question_length < 1:
            raise ValueError("max_question_length must be positive or None")

        self.blip_name = blip_name
        self.bert_name = bert_name
        self.blip = BlipForConditionalGeneration.from_pretrained(blip_name)
        self.bert = BertModel.from_pretrained(bert_name)
        self.question_tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.blip_processor = BlipProcessor.from_pretrained(blip_name)
        self.caption_tokenizer = self.blip_processor.tokenizer
        self.max_question_length = max_question_length

        vision_dim = int(self.blip.config.vision_config.hidden_size)
        question_dim = int(self.bert.config.hidden_size)
        decoder_dim = int(self.blip.config.text_config.hidden_size)
        if not (vision_dim == question_dim == decoder_dim):
            raise ValueError(
                "The paper concatenates question and image features directly. "
                "This implementation therefore requires equal ViT, BERT and "
                f"BLIP-decoder hidden sizes; got ViT={vision_dim}, "
                f"BERT={question_dim}, decoder={decoder_dim}. Do not insert "
                "unreported projection layers silently."
            )
        self.hidden_dim = vision_dim
        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"QAFD num_heads={num_heads} does not divide hidden_dim={self.hidden_dim}"
            )

        self.qafd = QAFD(
            dim=self.hidden_dim,
            num_heads=num_heads,
            alpha=alpha,
            num_layers=qafd_layers,
            ffn_ratio=qafd_ffn_ratio,
            ffn_activation=qafd_ffn_activation,
            ffn_residual=qafd_ffn_residual,
        )

        # Keep separate tokenizers: BERT for questions, BLIP for PET/captions.
        text_config = self.blip.config.text_config
        self.caption_bos_token_id = self._first_defined(
            self.caption_tokenizer.bos_token_id,
            getattr(text_config, "decoder_start_token_id", None),
            text_config.bos_token_id,
            self.caption_tokenizer.cls_token_id,
        )
        # BLIP commonly uses a BERT-derived tokenizer with SEP as EOS.
        self.caption_eos_token_id = self._first_defined(
            self.caption_tokenizer.eos_token_id,
            self.caption_tokenizer.sep_token_id,
            text_config.eos_token_id,
            getattr(text_config, "sep_token_id", None),
        )
        self.caption_pad_token_id = self._first_defined(
            self.caption_tokenizer.pad_token_id,
            self.blip.config.text_config.pad_token_id,
        )
        if any(
            token_id is None
            for token_id in (
                self.caption_bos_token_id,
                self.caption_eos_token_id,
                self.caption_pad_token_id,
            )
        ):
            raise ValueError(
                "Could not determine BLIP BOS/EOS/PAD token IDs from the BLIP "
                "tokenizer and decoder configuration."
            )

        decoder_vocab_size = int(self.blip.text_decoder.get_input_embeddings().num_embeddings)
        for token_name, token_id in (
            ("BOS", self.caption_bos_token_id),
            ("EOS", self.caption_eos_token_id),
            ("PAD", self.caption_pad_token_id),
        ):
            if not 0 <= int(token_id) < decoder_vocab_size:
                raise ValueError(
                    f"BLIP {token_name} token ID {token_id} is outside decoder "
                    f"vocabulary size {decoder_vocab_size}"
                )

        # The entire pretrained BLIP visual encoder and text decoder, plus BERT,
        # remain trainable. The paper describes fine-tuning, not freezing these
        # components in the reported method.
        self.last_qafd_attention: Optional[torch.Tensor] = None
        self.last_decoder_attention: Optional[torch.Tensor] = None
        self.last_decoder_attention_mask: Optional[torch.Tensor] = None
        self.last_caption_query_mask: Optional[torch.Tensor] = None
        self.last_encoder_attention_mask: Optional[torch.Tensor] = None

    @staticmethod
    def _first_defined(*values: Optional[int]) -> Optional[int]:
        for value in values:
            if value is not None:
                return int(value)
        return None

    def encode(
        self,
        pixel_values: torch.Tensor,
        questions: Union[str, Sequence[str]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode image/question and return decoder memory, mask and QAFD map."""
        if pixel_values.ndim != 4:
            raise ValueError(
                f"pixel_values must have shape [B, C, H, W], got {tuple(pixel_values.shape)}"
            )
        if isinstance(questions, str):
            questions = [questions]
        if len(questions) != pixel_values.shape[0]:
            raise ValueError(
                f"Received {len(questions)} questions for batch size {pixel_values.shape[0]}"
            )
        if any(not isinstance(question, str) or not question.strip() for question in questions):
            raise ValueError("Every question must be a non-empty string")

        target_device = device or pixel_values.device
        pixel_values = pixel_values.to(target_device)
        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self.max_question_length is not None:
            tokenizer_kwargs["max_length"] = self.max_question_length
        question_batch = self.question_tokenizer(list(questions), **tokenizer_kwargs)
        question_batch = question_batch.to(target_device)

        question_outputs = self.bert(
            input_ids=question_batch.input_ids,
            attention_mask=question_batch.attention_mask,
            return_dict=True,
        )
        question_features = question_outputs.last_hidden_state
        question_padding_mask = question_batch.attention_mask.eq(0)

        vision_outputs = self.blip.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        image_features = vision_outputs.last_hidden_state

        if image_features.shape[-1] != question_features.shape[-1]:
            raise RuntimeError(
                "ViT and BERT feature dimensions differ after model construction: "
                f"{image_features.shape[-1]} vs {question_features.shape[-1]}"
            )

        distilled_image, question_aware, qafd_attention = self.qafd(
            question_tokens=question_features,
            image_tokens=image_features,
            question_padding_mask=question_padding_mask,
        )

        # Figure 5 / Section 3.3.3: concatenate F_tilde_q and F_tilde_v.
        decoder_memory = torch.cat([question_aware, distilled_image], dim=1)
        image_memory_mask = torch.zeros(
            distilled_image.shape[:2], dtype=torch.bool, device=target_device
        )
        memory_padding_mask = torch.cat(
            [question_padding_mask, image_memory_mask], dim=1
        )
        self.last_encoder_attention_mask = (~memory_padding_mask).long()
        return decoder_memory, memory_padding_mask, qafd_attention

    def forward(
        self,
        pixel_values: torch.Tensor,
        questions: Union[str, Sequence[str]],
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run teacher-forced PET-conditioned caption generation.

        ``decoder_input_ids`` must already contain [BOS] + PET prompt + caption
        prefix tokens, as produced by the paired QACapDataset. ``labels`` are
        optionally supplied only to identify the caption-token positions used
        by QARCL; the paper's L_cap is calculated externally by the loss module.
        """
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must have shape [B, decoder_length]")
        if decoder_input_ids.shape[0] != pixel_values.shape[0]:
            raise ValueError("Decoder-input batch size does not match image batch size")
        decoder_input_ids = decoder_input_ids.to(pixel_values.device, dtype=torch.long)
        if labels is not None:
            if labels.shape != decoder_input_ids.shape:
                raise ValueError("labels must have the same shape as decoder_input_ids")
            labels = labels.to(pixel_values.device, dtype=torch.long)
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_ids.ne(self.caption_pad_token_id).long()
        else:
            if decoder_attention_mask.shape != decoder_input_ids.shape:
                raise ValueError("decoder_attention_mask must match decoder_input_ids shape")
            decoder_attention_mask = decoder_attention_mask.to(
                pixel_values.device, dtype=torch.long
            )

        memory, memory_padding_mask, qafd_attention = self.encode(
            pixel_values=pixel_values,
            questions=questions,
            device=pixel_values.device,
        )
        if decoder_input_ids.numel() and (
            int(decoder_input_ids.min()) < 0
            or int(decoder_input_ids.max()) >= self.blip.text_decoder.get_input_embeddings().num_embeddings
        ):
            raise ValueError("decoder_input_ids contain token IDs outside the BLIP vocabulary")

        max_positions = int(self.blip.config.text_config.max_position_embeddings)
        if decoder_input_ids.shape[1] > max_positions:
            raise ValueError(
                f"Decoder sequence length {decoder_input_ids.shape[1]} exceeds BLIP "
                f"max_position_embeddings={max_positions}; do not truncate the PET prompt silently"
            )

        decoder_outputs = self.blip.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=(~memory_padding_mask).long(),
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        logits = getattr(decoder_outputs, "logits", None)
        if logits is None:
            raise RuntimeError("BLIP text decoder did not return vocabulary logits")

        # QARCL requires the last decoder cross-attention map. Never silently
        # continue without it: that would train a different objective.
        cross_attentions = getattr(decoder_outputs, "cross_attentions", None)
        if not isinstance(cross_attentions, (tuple, list)) or not cross_attentions:
            raise RuntimeError(
                "BLIP text decoder did not return cross_attentions even though "
                "output_attentions=True. QARCL cannot be computed. Check the "
                "Transformers version and BLIP decoder configuration; do not "
                "continue paper-reproduction training with QARCL disabled."
            )
        final_cross_attention = cross_attentions[-1]
        if final_cross_attention is None or final_cross_attention.ndim != 4:
            raise RuntimeError(
                "Expected final BLIP cross-attention with shape "
                "[B, heads, decoder_tokens, memory_tokens]"
            )

        # Average heads to obtain the 2-D attention map described in the paper,
        # then retain only its image-region columns (the memory is [F_tilde_q; F_tilde_v]).
        decoder_attention = final_cross_attention.mean(dim=1)
        question_token_count = qafd_attention.shape[1]
        image_token_count = qafd_attention.shape[2]
        expected_memory_length = question_token_count + image_token_count
        if decoder_attention.shape[-1] != expected_memory_length:
            raise RuntimeError(
                "Decoder cross-attention memory length does not match the "
                "[question; image] memory sequence: "
                f"got {decoder_attention.shape[-1]}, expected {expected_memory_length}"
            )
        caption_image_attention = decoder_attention[..., question_token_count:]
        if caption_image_attention.shape[-1] != image_token_count:
            raise RuntimeError("QARCL image-region attention dimensions do not match")

        self.last_qafd_attention = qafd_attention
        self.last_decoder_attention = caption_image_attention
        self.last_decoder_attention_mask = decoder_attention_mask
        # QARCL is intended to compare the regions used while generating the
        # caption, not the PET instruction tokens. data_updated.py marks PET
        # and BOS positions with -100 in labels, so retain that mask for losses.py.
        self.last_caption_query_mask = (
            labels.ne(-100) if labels is not None else None
        )
        return logits, qafd_attention

    @torch.inference_mode()
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        questions: Union[str, Sequence[str]],
        max_new_tokens: int = 64,
    ) -> List[str]:
        """Generate question-aware captions using the same PET prefix as training.

        The paper specifies PET-conditioned autoregressive caption generation,
        but its main text does not report a decoding strategy or maximum output
        length. This helper uses greedy decoding as an explicit, replaceable
        inference default; it does not claim that greedy decoding is paper-reported.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if isinstance(questions, str):
            questions = [questions]
        if len(questions) != pixel_values.shape[0]:
            raise ValueError("Number of questions must equal image batch size")

        memory, memory_padding_mask, _ = self.encode(
            pixel_values=pixel_values,
            questions=questions,
            device=pixel_values.device,
        )
        memory_attention_mask = (~memory_padding_mask).long()
        tokenizer = self.caption_tokenizer
        outputs: List[str] = []
        max_positions = int(self.blip.config.text_config.max_position_embeddings)

        # Generate per example to avoid inserting right-padding between a PET
        # prefix and its autoregressively generated caption tokens.
        for index, question in enumerate(questions):
            pet_prompt = PET_PROMPT_TEMPLATE.format(question=question)
            prompt_ids = tokenizer.encode(pet_prompt, add_special_tokens=False)
            sequence_ids = [int(self.caption_bos_token_id)] + [int(x) for x in prompt_ids]
            if len(sequence_ids) >= max_positions:
                raise ValueError(
                    f"PET prompt for sample {index} uses {len(sequence_ids)} tokens, "
                    f"leaving no room within BLIP max_position_embeddings={max_positions}"
                )

            generated = torch.tensor(
                [sequence_ids], dtype=torch.long, device=pixel_values.device
            )
            sample_memory = memory[index : index + 1]
            sample_memory_mask = memory_attention_mask[index : index + 1]

            for _ in range(max_new_tokens):
                decoder_outputs = self.blip.text_decoder(
                    input_ids=generated,
                    attention_mask=torch.ones_like(generated, dtype=torch.long),
                    encoder_hidden_states=sample_memory,
                    encoder_attention_mask=sample_memory_mask,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                )
                next_token = decoder_outputs.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )
                generated = torch.cat([generated, next_token], dim=1)
                if int(next_token.item()) == int(self.caption_eos_token_id):
                    break
                if generated.shape[1] >= max_positions:
                    break

            # Decode only generated caption tokens, never the PET instruction.
            new_tokens = generated[0, len(sequence_ids) :]
            outputs.append(
                tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )
        return outputs

    def get_last_qarcl_attention(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw QAFD and caption-to-image attention maps for the JS loss.

        Shapes:
            QAFD:    [B, question_tokens, image_tokens]
            Caption: [B, decoder_tokens, image_tokens]

        The loss module should perform the paper's region-wise aggregation and
        JS divergence. The two query axes need not have equal lengths.
        """
        if self.last_qafd_attention is None or self.last_decoder_attention is None:
            raise RuntimeError(
                "QARCL attention maps are unavailable. Run a forward pass with "
                "output_attentions enabled first."
            )
        return self.last_qafd_attention, self.last_decoder_attention
