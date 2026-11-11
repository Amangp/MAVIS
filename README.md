# QACap — research-grounded implementation scaffold

This repository is a reconstruction of the method described in Zhen Yang et al.,
*Separation of Powers: On Segregating Knowledge from Observation in LLM-enabled
Knowledge-based Visual Question Answering*, CVPR 2025.

## Reproduction status — read before training

This is **not yet an exact reproduction** of the authors' experiment. The attached
10-page main paper omits settings and prompts that it explicitly delegates to the
supplementary material. The current project contains transparent implementation
choices for some of those gaps, and those choices must not be reported as author-
confirmed. See [`AUDIT_REPORT.md`](AUDIT_REPORT.md) and
[`REPRODUCTION_GAPS.md`](REPRODUCTION_GAPS.md).

The main paper specifies the core pipeline: BLIP-based visual captioning, ViT image
features, BERT question features, QAFD, PET question-prefix conditioning, a caption
decoder that attends to question/image tokens, QARCL using JS divergence, and the
joint loss `L = L_cap + lambda_JS * L_JS`. It also reports sequential fine-tuning:
PromptCap for 30 epochs, then QACap for 50 epochs, with initial LR `2e-5` in both
stages. The paper reports 29,569 QACap examples derived from OK-VQA, A-OKVQA and
FVQA, with five human COCO captions, three GPT-4 candidate captions, refinement,
answer-helpfulness scoring and CLIP tie-breaking.

**Important unresolved point:** Figure 5 depicts a Feed Forward block in QAFD,
whereas the present QAFD implementation does not include one. The main-text
mathematical description does not fully specify that block. Do not treat the current
architecture as fully figure-identical until this is reconciled with supplementary
material or author code.

## Repository layout

- `qacap/model.py`: BLIP/ViT + BERT, QAFD, PET-conditioned decoder and attention maps.
- `qacap/losses.py`: caption NLL and QARCL JS divergence.
- `qacap/data.py`: JSONL triplet dataset and BLIP image preprocessing.
- `qacap/metrics.py`: answer normalization and paper-described metric helpers.
- `qacap/llm.py`: Claude 3.5 prompt/API adapter.
- `qacap/checkpoint.py`: atomic resumable checkpoints.
- `scripts/train.py`: unified PromptCap-to-QACap two-stage trainer.
- `scripts/generate_qacap_candidates.py`: GPT-4 caption generation/refinement and selection.
- `scripts/prepare_data.py`: validates a completed, selected-caption manifest.
- `scripts/caption.py`, `scripts/answer.py`: single-example inference.
- `tests/`: QAFD and loss unit tests.

## Environment

Use a Python version supported by the selected PyTorch/Transformers builds. The
project requirements specify minimum versions rather than a fully tested lockfile;
for a reproducible experiment, record and pin the exact versions and model revisions
that you actually run. Install a PyTorch wheel matching your CUDA runtime from
https://pytorch.org/get-started/locally/ before installing the remaining packages.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m pytest -q
```

The current audit environment could not instantiate BLIP/BERT because Transformers
and the external API/CLIP dependencies were not installed. The passing unit tests
therefore validate only QAFD/loss logic with local tensors, not a full pretrained
model forward pass or GPU training run.

## Data preparation — source adapters are still missing

The training dataset loader expects JSONL with one example per line:

```json
{"image":"images/000000000139.jpg","question":"What toy is this?","caption":"A boy holds a stuffed animal in his backpack.","answers":["stuffed animal","plush toy"],"image_id":139,"source_dataset":"okvqa"}
```

Relative image paths are resolved under `data.root`. `scripts/prepare_data.py` does
**not** convert or merge the original OK-VQA, A-OKVQA, FVQA, COCO or PromptCap source
formats. That source-specific extraction/merge step must be built and verified
against the downloaded dataset versions before candidate generation. It must use
training splits only and preserve question/image IDs and source provenance.

For the QACap synthetic dataset, the candidate-generation input additionally needs
`coco_captions` containing exactly five human COCO captions per image-question row.
The intended sequence is:

1. Build a source manifest from the official KBVQA training splits and the five
   corresponding human-written COCO captions.
2. Generate/refine/score/select captions with `scripts/generate_qacap_candidates.py`.
   This script needs OpenAI GPT-4, Anthropic Claude 3.5 and a configured CLIP model;
   API model IDs, token budgets and the CLIP checkpoint must be supplied explicitly.
3. Validate the selected manifest with `scripts/prepare_data.py` and
   `--require-candidate-selection --expected-count 29569`.
4. Prepare the PromptCap training manifest separately. The current project does not
   include a PromptCap source-format importer or official split mapper.

The candidate prompts and phrase filter in this project are reconstructions from the
main-paper description, not verified verbatim supplementary prompts. Human review
for answer leakage and unsupported visual content remains necessary.

## Training

The current trainer runs both stages sequentially in one process, reusing the same
model and optimizer objects. From the repository root:

```bash
# Smoke test only — 1 epoch per stage, tiny subset; NOT a paper experiment.
python scripts/train.py --config configs/debug.yaml --allow-non-paper-settings

# Full stated schedule — only after preparing manifests and verifying unknown settings.
python scripts/train.py --config configs/paper_reproduction.yaml

# Resume from the latest epoch-boundary checkpoint.
python scripts/train.py --config configs/paper_reproduction.yaml --resume outputs/qacap/last.pt
```

The full config preserves the main-text 30 + 50 epoch schedule and LR `2e-5`, but
currently contains unverified values for `alpha`, `lambda_js`, `num_heads`, and
`qafd_layers`. The trainer prints some of these as unconfirmed but does not establish
their correctness. The QARCL loss is currently applied in both phases; phase-specific
loss treatment should be verified from the supplementary material. Do not call a
run an exact reproduction until these items are resolved.

With the supplied config, no caption-supervised validation manifest is configured,
so a `best.pt` is not produced. The final checkpoint is `outputs/qacap/final.pt`
and the latest resumable checkpoint is `outputs/qacap/last.pt`. If validation is
configured, phase-specific best checkpoints are saved under
`outputs/qacap/<phase>/best.pt`; the current trainer's validation is caption-loss
validation, not the paper's end-to-end VQA benchmark evaluation.

## Inference

The paper's main text does not disclose the caption decoding length or an exact
Claude API snapshot/token budget. Configure `inference.max_new_tokens`,
`inference.llm_model`, and `inference.llm_max_tokens` in the training YAML before
training, or pass the corresponding CLI/environment settings at inference.

```bash
python scripts/caption.py --checkpoint outputs/qacap/final.pt \
  --image path/to/image.jpg --question "What toy is this?" --max-new-tokens 64

python scripts/answer.py --checkpoint outputs/qacap/final.pt \
  --image path/to/image.jpg --question "What toy is this?" \
  --max-new-tokens 64 --llm-model "$QACAP_LLM_MODEL" \
  --llm-max-tokens "$QACAP_LLM_MAX_TOKENS"
```

Set `ANTHROPIC_API_KEY` securely in the environment. Never commit API keys. The
numbers `64` and any token budget are implementation choices unless verified from
supplementary material.

## Evaluation and ablations — not yet complete

`qacap/metrics.py` contains metric helpers, but there is no complete benchmark runner
that loads official OK-VQA/A-OKVQA validation splits, generates the two reported
answer formats (one-word/phrase and open-ended), parses A-OKVQA multiple-choice
options, invokes Claude consistently, and exports the paper's result tables. The
current `scripts/answer.py` handles one sample and one answer prompt only. Ablation
switches/runners for baseline, QAFD, QARCL and PET are also not implemented.

These are required before claiming full experimental reproduction. See
`AUDIT_REPORT.md` for the exact gaps and recommended order of work.
