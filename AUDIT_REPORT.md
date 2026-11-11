# QACap project audit report

**Project:** `QACap_Research_Implementation(1).zip`  
**Target paper:** Zhen Yang et al., *Separation of Powers: On Segregating Knowledge
from Observation in LLM-enabled Knowledge-based Visual Question Answering*, CVPR 2025.  
**Audit basis:** the ZIP project, its configs/tests, and the 10-page paper PDF supplied
in the workspace.  
**Audit status:** static and unit-test audit completed; pretrained-model integration
and end-to-end reproduction remain unverified.

## Executive result

The repository is a useful modular scaffold and its 22 current unit tests pass, but
it is **not yet a complete or exact reproduction** of the CVPR 2025 experiment.
There are three kinds of blockers:

1. **Confirmed project defects:** stale README commands/checkpoint paths and missing
   package-root setup in direct training/answer script invocation.
2. **Missing required workflows:** raw dataset import/merge, full benchmark evaluation
   for all paper tasks, and ablation runners.
3. **Paper details not recoverable from the supplied main paper:** `alpha`, `lambda_JS`,
   QAFD layer/head settings, QAFD FFN internals, exact data prompts/filter list, and
   training/API implementation details.

The audited copy fixes the first category and documents the others. It intentionally
 does not invent the unknown experimental values or silently change the model to match
an under-specified figure block.

## 1. Paper-to-code alignment matrix

| Paper component | Present implementation | Audit judgment |
|---|---|---|
| Image encoder: pretrained ViT | BLIP vision model `last_hidden_state` | Broadly aligned; exact checkpoint/revision must be verified |
| Question encoder: BERT tokenizer/encoder | `BertTokenizer` + `BertModel` | Aligned at component level; exact checkpoint unknown in main text |
| QAFD question MHSA | `nn.MultiheadAttention` | Aligned at operation level |
| QAFD question-to-image MHCA | `nn.MultiheadAttention` with question queries/image keys-values | Aligned at operation level |
| Token-wise regional score | Max over question-token axis | Matches Eq. (3) |
| Sentence-wise score | Max-pool question features then cosine with image tokens | Matches Eqs. (4)-(5) |
| Similarity blend/reweight/residual | Mean of scores; elementwise multiply; alpha blend | Matches Eqs. (6)-(8), but alpha is unverified |
| QAFD Feed Forward | Not implemented in `_QAFDBlock` | **Figure 5 discrepancy**; dimensions/activation not stated in main-text equations |
| PET | Question prefixed with paper example prompt | Aligned in concept; sequence/target masking is implementation-specific |
| Caption generation | BLIP text decoder cross-attends to `[question; image]` memory | Broadly aligned; exact decoder arrangement and FFN details need runtime/source check |
| QARCL | Last QAFD and last decoder cross-attention; head mean; query max-pool; region JS | Diagram-consistent reconstruction; verify aggregation and masking details |
| Joint objective | Caption NLL + lambda times JS | Aligned with Eq. (12); lambda unknown |
| Stage 1/2 schedule | PromptCap 30 epochs, QACap 50 epochs, LR 2e-5 | Matches main-text schedule |
| Stage-specific QARCL behavior | QARCL currently active in both phases | Not explicitly confirmed by main text |
| QACap dataset sources/count | Candidate pipeline and manifest validator | Candidate logic present; source-specific import/merge absent; 29,569 total can be asserted only after data validation |
| Candidate generation | GPT-4 creates three captions; separate refinement | Conceptually aligned; exact prompts are reconstructed, not verbatim |
| Helpfulness selection | Claude response checked for annotated answers in order | Close to Section 3.2.2; exact scoring normalization not verified |
| Truthfulness selection | CLIP image-caption cosine; highest among helpfulness ties | Aligned conceptually; exact CLIP checkpoint unknown |
| LLM answer stage | Claude adapter with Task Instruction / Context / Question | Figure 2 prompt structure represented; API snapshot/token settings unknown |
| OK-VQA evaluation | Soft accuracy helper only | Metric helper present; no full benchmark execution |
| A-OKVQA direct-answer | Ordered phrase helper | Partial helper only; no benchmark runner or official source loader |
| A-OKVQA multiple-choice | Exact normalized choice helper | Partial helper only; no option parsing/runner |
| Ablation studies | No switches/runner for all variants | Missing |

## 2. Confirmed code issues and changes applied

### 2.1 Direct script import path

`python scripts/train.py` and `python scripts/answer.py` launch with the `scripts/`
directory at `sys.path[0]`. Neither file originally inserted the repository root,
so dynamic imports such as `from qacap.model import QACap` could fail when using the
README's direct-script form. The audited copy adds the same repository-root bootstrap
already present in `caption.py` and `generate_qacap_candidates.py`.

### 2.2 README trainer mismatch

The README originally used `--stage promptcap` and `--stage qacap`; the current parser
has no `--stage` argument. It runs PromptCap then QACap automatically in one invocation.
The README now uses the actual interface.

### 2.3 Checkpoint output mismatch

With the supplied config there is no validation manifest. The trainer writes
`outputs/qacap/last.pt` and copies the final checkpoint to `outputs/qacap/final.pt`.
It does not produce a root-level `outputs/qacap/best.pt`. The README now reflects the
actual paths and explains phase-specific best checkpoints are only created if a
caption-supervised validation manifest is configured.

### 2.4 No benchmark evaluator

`qacap/metrics.py` provides isolated metric functions. It does not load official
validation JSON, run the LLM in both answer formats, parse A-OKVQA answer choices,
aggregate results, or output Table 1/Table 2 comparisons. `scripts/answer.py` is a
single-example interface, not a benchmark runner.

## 3. Method and dataset gaps requiring decisions

### 3.1 QAFD Feed Forward

Figure 5 depicts a Feed Forward stage after QAFD cross-attention. The current block
returns the raw MHCA output. Eq. (2) in the main text defines the cross-attention output
but does not explain this displayed FFN's width, activation, residual or normalization.
This is a real architecture-to-figure gap. Adding a conventional `4d` GELU MLP would
be a plausible implementation but would still be an assumption, not a verified paper
specification. It is therefore left unresolved in this audited copy.

### 3.2 Unverified numerical settings

The current config has values such as `alpha: 0.5`, `lambda_js: 0.1`, `num_heads: 12`
and `qafd_layers: 1`. These are implementation values, not values established by the
supplied main paper. The trainer prints some values as unconfirmed, but it does not
prove them. An experiment using them should be labelled a **reconstruction with
assumptions**, not an exact reproduction.

### 3.3 Dataset source builder

`generate_qacap_candidates.py` begins from a row that already contains image path,
question, answer annotations and five COCO captions. `prepare_data.py` validates the
finished manifest. Neither converts/joins original OK-VQA, A-OKVQA, FVQA, COCO and
PromptCap files. A robust adapter must preserve source IDs/splits, keep only training
questions in the 29,569 QACap training construction, associate five correct COCO
captions, deduplicate and report source counts.

### 3.4 Caption generation and scoring prompts

The main paper describes the sequence but refers to supplementary material for full
prompts/refinement cases. The project uses reconstructed prompts and only the two
phrase examples named in the main text as default blocked phrases. That does not
establish exact reproduction of the authors' GPT-4 annotation pipeline.

### 3.5 Evaluation protocol

The paper describes one-word/phrase answers judged with traditional exact matching,
open-ended answer evaluation based on whether reference text appears in the generated
response, OK-VQA soft accuracy, and separate A-OKVQA direct-answer/multiple-choice
results. The current code contains helpers but no integrated evaluation workflow or
both answer-mode prompts. It is insufficient to validate the reported 68.2%, 66.3%
and 76.7% results.

### 3.6 Real model compatibility

`qacap/model.py` depends on Hugging Face BLIP output semantics, especially the final
`cross_attentions` tensor and its image-token columns. The current environment lacks
Transformers, so that behavior was not exercised. A version-pinned smoke test must
instantiate the exact BLIP/BERT checkpoints, run a forward/backward pass, assert all
attention shapes/finite values, and verify a checkpoint reload before committing to
a long training run.

## 4. What is tested

| Check | Result |
|---|---|
| Python syntax compilation (`python -m compileall -q qacap scripts tests`) | PASS |
| Existing QAFD/loss tests (`python -m pytest -q`) | 22 passed |
| Standalone `pytest -q` | PASS after adding `pytest.ini` (`pythonpath = .`) |
| BLIP/BERT model construction/forward/backward | NOT RUN — Transformers absent |
| CLIP scoring and GPT-4/Claude API | NOT RUN — packages/credentials absent |
| GPU training, resume under CUDA, benchmark evaluation | NOT RUN |

The container had PyTorch `2.10.0+cpu`, Python `3.13`, Pillow `12.3.0` and PyYAML
`6.0.3`; it did not have `transformers`, `timm`, `open_clip_torch`, `anthropic` or
`openai`. No claim of pretrained-model runtime correctness is made.

## 5. Recommended completion order

1. Obtain the paper supplementary PDF/authors' config. Resolve QAFD FFN and numeric
   architecture/loss values first.
2. Build and unit-test raw source adapters for PromptCap, OK-VQA, A-OKVQA, FVQA and
   COCO. Prove the exact 29,569 training records and preserve no validation/test leakage.
3. Pin PyTorch, Transformers, model revisions and CUDA environment; run model forward,
   backward, checkpoint save/reload and one-batch train smoke test.
4. Verify QARCL attention map extraction against the exact Transformers BLIP release;
   confirm whether QARCL is active in both training stages.
5. Implement complete OK-VQA and A-OKVQA benchmark evaluation, both answer styles,
   MC option parsing, official answer normalization, and machine-readable prediction logs.
6. Add ablation configurations for baseline/QAFD/QARCL/PET and test that each ablation
   disables only the intended paper component.
7. Run full 30 + 50 schedule and report results only after evaluation and reproducibility
   metadata are available.

## 6. Reference results in the supplied main paper

- OK-VQA validation: 68.2% for QACap + Claude 3.5.
- A-OKVQA validation: 66.3% direct answer and 76.7% multiple choice; the table reports
  a 71.5% average.

These are target reference values, not expected guarantees for this reconstruction.
