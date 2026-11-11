# Dataset preparation notes

The paper uses training data derived from:
- OK-VQA
- A-OKVQA
- FVQA
- MS COCO images and the five human-written COCO captions
- PromptCap training data for stage one

The paper reports 29,569 QACap image-question-caption training entries. It describes
three GPT-4 candidate captions per example, then selection by helpfulness (LLM answer
score against ground-truth answers) and truthness (CLIP image-caption similarity).
The final caption is selected among candidates with the strongest helpfulness score
using CLIP similarity to break ties.

This repository does not include or redistribute these datasets. Download the original
datasets from their official project pages and preserve their train/validation/test splits.
Do not mix validation answers into training candidate generation. Store image paths and
metadata in JSONL. Record exact image IDs and annotation versions.
