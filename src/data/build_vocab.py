import json
import pickle
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

# This file is expected at D:\MAVIS\src\data\build_vocab.py
ROOT = Path(__file__).resolve().parents[2]

QUESTIONS_FILE = (
    ROOT / "datasets" / "v2_Questions_Train_mscoco"
    / "v2_OpenEnded_mscoco_train2014_questions.json"
)
ANNOTATIONS_FILE = (
    ROOT / "datasets" / "v2_Annotations_Train_mscoco"
    / "v2_mscoco_train2014_annotations.json"
)
OUTPUT_DIR = ROOT / "vocab"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MIN_WORD_FREQ = 1
MAX_ANSWER_VOCAB = 7500


def normalize_question(text: str) -> str:
    """Keep question tokenization consistent with MAVIS data loading."""
    text = text.lower().strip()
    text = re.sub(r"([?.!,;:])", r" \1 ", text)
    return re.sub(r"\s+", " ", text)


def normalize_answer(text: str) -> str:
    """Minimal normalization; official VQA scoring uses richer rules."""
    return " ".join(str(text).lower().strip().split())


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find dataset file:\n{path}\n"
            "Verify that your VQA v2 dataset is inside D:\\MAVIS\\datasets."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def backup_if_exists(path: Path):
    """Create one timestamped backup and never overwrite an earlier backup."""
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")


def main():
    print(f"Project root: {ROOT}")
    print("Loading VQA v2 training JSON files...")

    questions_data = read_json(QUESTIONS_FILE)
    annotations_data = read_json(ANNOTATIONS_FILE)
    questions = questions_data.get("questions", [])
    annotations = annotations_data.get("annotations", [])

    if not questions or not annotations:
        raise ValueError("Question or annotation list is empty.")

    annotations_by_qid = {}
    for annotation in annotations:
        qid = annotation.get("question_id")
        if qid is not None:
            annotations_by_qid[qid] = annotation

    question_counter = Counter()
    answer_counter = Counter()
    matched = 0
    missing = 0

    for question_record in questions:
        question = question_record.get("question", "")
        if question:
            question_counter.update(normalize_question(question).split())

        qid = question_record.get("question_id")
        annotation = annotations_by_qid.get(qid)
        if annotation is None:
            missing += 1
            continue
        matched += 1

        # Count all human reference answers, not only multiple_choice_answer.
        for answer_record in annotation.get("answers", []):
            if isinstance(answer_record, dict):               answer = answer_record.get("answer", "")
            else:
                answer = answer_record
            answer = normalize_answer(answer)
            if answer:
                answer_counter[answer] += 1

    if matched == 0:
        raise ValueError("No matching question/annotation IDs were found.")

    word_items = [
        (word, count)
        for word, count in question_counter.items()
        if count >= MIN_WORD_FREQ and word not in {PAD_TOKEN, UNK_TOKEN}
    ]
    word_items.sort(key=lambda item: (-item[1], item[0]))
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in word_items:
        word2idx[word] = len(word2idx)

    answer_items = sorted(
        answer_counter.items(), key=lambda item: (-item[1], item[0])
    )
    if MAX_ANSWER_VOCAB is not None:
        answer_items = answer_items[:MAX_ANSWER_VOCAB]
    ans2idx = {answer: index for index, (answer, _) in enumerate(answer_items)}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    word_path = OUTPUT_DIR / "word2idx.pkl"
    answer_path = OUTPUT_DIR / "ans2idx.pkl"
    stats_path = OUTPUT_DIR / "vocab_stats.json"

    # Back up current artifacts before overwriting. The old word vocabulary
    # should remain identical, but we back it up for reproducibility.
    backup_if_exists(word_path)
    backup_if_exists(answer_path)
    backup_if_exists(stats_path)

    with word_path.open("wb") as handle:
        pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with answer_path.open("wb") as handle:
        pickle.dump(ans2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    stats = {
        "source_questions": str(QUESTIONS_FILE.relative_to(ROOT)),
        "source_annotations": str(ANNOTATIONS_FILE.relative_to(ROOT)),
        "question_records": len(questions),
        "annotation_records": len(annotations),
        "matched_question_annotation_pairs": matched,
        "questions_without_annotation": missing,
        "question_vocab_size_including_special_tokens": len(word2idx),
        "answer_vocab_size": len(ans2idx),
        "min_word_freq": MIN_WORD_FREQ,
        "max_answer_vocab": MAX_ANSWER_VOCAB,
        "answer_normalization": "lowercase_and_collapse_whitespace",
        "answer_vocabulary_ranked_by": "descending_frequency_then_alphabetical",
    }
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print("\nVocabulary build complete.")
    print(f"Matched training pairs: {matched:,}")
    print(f"Question vocabulary: {len(word2idx):,} tokens")
    print(f"Answer vocabulary: {len(ans2idx):,} classes")
    print(f"Saved: {word_path}")
    print(f"Saved: {answer_path}")
    print(f"Stats: {stats_path}")
    print(f"Sample question token: 'what' -> {word2idx.get('what', word2idx[UNK_TOKEN])}")
    print(f"Sample answer class: 'net' -> {ans2idx.get('net', 'not in vocabulary')}")
    print("\nIMPORTANT: update/check the model classifier output dimension and all target mappings.")
    print("This script does not modify your model architecture or training code.")


if __name__ == "__main__":
    main()
