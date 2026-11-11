# Reproduction gaps and decisions required

This file is an actionable record of what is not established by the provided
10-page CVPR 2025 main paper. Do not silently fill these gaps with guessed values.

## Architecture gaps

1. **QAFD Feed Forward block:** Figure 5 draws a Feed Forward block after the
   question-to-image cross-attention inside QAFD. `qacap/model.py` currently omits
   it. Eq. (2) defines `F_tilde_q = MHCA(...)` but the main text does not give the
   extra block's hidden width, activation, residual or normalization details.
   Resolve from supplement/authors before choosing a final implementation.
2. The numeric `alpha` in Eq. (8) is absent from the main text.
3. `lambda_JS` is explicitly delegated to supplementary material.
4. The QAFD repeat count `N`, attention head count and all FFN details are not
   specified numerically in the main paper body.
5. Confirm the exact BLIP and BERT pretrained checkpoint variants/revisions and
   confirm that the chosen Transformers release returns the expected final
   decoder cross-attention weights and uses the expected encoder memory mask.
6. Confirm how attention heads/query tokens are aggregated before the JS loss.
   The current implementation averages heads and max-pools over query tokens,
   consistent with Figure 5's diagram, but those mechanics should be checked
   against supplementary details/author code.

## Training gaps

7. The main text specifies sequential training: PromptCap 30 epochs, then QACap
   50 epochs, initial LR `2e-5` in each phase. The actual optimizer, weight decay,
   scheduler, batch size, clipping, augmentation, precision and seeds are not
   specified in the body.
8. The current trainer applies QARCL in both PromptCap and QACap phases. The main
   text does not state clearly whether the regional consistency term is applied
   during the first phase; verify this rather than treating the current behavior
   as author-confirmed.
9. The current trainer uses AdamW with configured weight decay. This is an
   implementation choice, not a main-text-confirmed optimizer setup.
10. The current validation path calculates caption-supervised loss only. It is not
    the paper's OK-VQA/A-OKVQA end-to-end VQA evaluation.

## Dataset gaps

11. There is no source-specific importer/merger for the original OK-VQA,
    A-OKVQA and FVQA question/answer files, COCO captions, or PromptCap data.
    `prepare_data.py` validates a prebuilt manifest; it does not create one.
12. The exact 29,569 row construction, source-specific counts, deduplication,
    annotation versions and split mapping must be verified. Validation/test
    samples must never enter candidate generation or QACap training.
13. The initial GPT-4 and refinement prompts are reconstructed from the main
    paper description. The exact supplementary prompts and full phrase blacklist
    are not available in the uploaded paper. The current blocked phrase defaults
    include only the two examples in the paper body (`known for`, `demonstrating`).
14. The exact GPT-4 API snapshot, generation settings, Claude snapshot and CLIP
    checkpoint are not disclosed in the main paper body.
15. The current candidate helpfulness rule counts answer phrases occurring in
    order in the Claude response. Verify this normalization/matching behavior
    against supplementary implementation details if available.

## Evaluation and reproducibility gaps

16. No full benchmark evaluation script exists for OK-VQA, A-OKVQA direct-answer
    and A-OKVQA multiple-choice evaluation.
17. The main paper reports two answer types: short one-word/phrase answers and
    open-ended answers. The current `answer.py` uses one generic answering prompt
    and does not execute both evaluation modes.
18. No A-OKVQA choice parsing/option-conditioned inference exists. The current
    multiple-choice metric expects already-decoded choices.
19. No ablation runner/configuration is implemented for the paper's baseline,
    QAFD, QARCL and PET combinations.
20. The repository has no tested environment lockfile or pinned Hugging Face model
    revisions. Pin exact versions/revisions and store run metadata for a repeatable
    experiment.
21. Existing tests cover QAFD tensor behavior and losses (22 tests). They do not
    integration-test the BLIP/BERT model forward, dataset tokenizer/label alignment,
    checkpoint resume, GPT/Claude/CLIP calls, benchmark data loaders or GPU training.

## Project integration corrections applied in the audited copy

- Added repository-root import setup to `scripts/train.py` and `scripts/answer.py`
  so direct invocations documented in the README can find the `qacap` package.
- Rewrote README commands to match the actual unified trainer (no unsupported
  `--stage` option).
- Corrected checkpoint paths and explained that no `best.pt` is written unless a
  caption-supervised validation manifest is configured.
- Added explicit limitations for source data conversion and benchmark evaluation.

## Test evidence for this audited copy

- `python -m compileall -q qacap scripts tests`: passed.
- `python -m pytest -q`: 22 passed.
- Full model/training runtime: **not verified** in the audit environment because
  `transformers`, `timm`, `open_clip_torch`, `anthropic`, and `openai` are absent;
  PyTorch available here is CPU-only. The tests do not load pretrained checkpoints.
