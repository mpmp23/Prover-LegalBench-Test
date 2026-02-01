import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple

@dataclass(frozen=True)
class TaskConfig:
    name: str
    # If provided, constrain answers to these canonical labels.
    labels: List[str]
    # Mapping from model output -> canonical label
    normalizers: List[Tuple[Pattern[str], str]]
    # A short instruction for the model.
    instruction: str

    def normalize(self, s: str) -> Optional[str]:
        s_clean = (s or "").strip()
        # Common cleanup: take first line, strip punctuation.
        s_clean = s_clean.splitlines()[0].strip()
        s_clean = re.sub(r"[\s\t]+", " ", s_clean)
        s_clean = s_clean.strip(" .,:;\"'`()[]{}")
        low = s_clean.lower()

        # Exact match on canonical labels first (case-insensitive)
        for lab in self.labels:
            if low == lab.lower():
                return lab

        # Try regex normalizers
        for pat, lab in self.normalizers:
            if pat.search(low):
                return lab

        return None


# Recommended starter tasks (auto-gradable, mostly short outputs):
# - hearsay: binary classification (hearsay vs not hearsay)
# - personal_jurisdiction: binary classification (yes/no)
# - proa: binary classification (contains private right of action vs not)
# - privacy_policy_entailment: binary classification (Correct/Incorrect)
# - insurance_policy_interpretation: 3-way classification (Yes/No/Ambiguous)

TASKS: Dict[str, TaskConfig] = {
    "hearsay": TaskConfig(
    name="hearsay",
    labels=["Yes", "No"],
    normalizers=[
        (re.compile(r"\byes\b"), "Yes"),
        (re.compile(r"\bno\b"), "No"),
        (re.compile(r"\bhearsay\b"), "Yes"),
        (re.compile(r"\bnot\s+hearsay\b"), "No"),
        (re.compile(r"\b(inadmissible|out[- ]of[- ]court)\b"), "Yes"),
        (re.compile(r"\b(admissible|non[- ]hearsay)\b"), "No"),
    ],
    instruction=(
        "Is the evidence hearsay? Answer with exactly: 'Yes' or 'No'."
    ),
),
    "personal_jurisdiction": TaskConfig(
        name="personal_jurisdiction",
        labels=["Yes", "No"],
        normalizers=[
            (re.compile(r"\byes\b"), "Yes"),
            (re.compile(r"\bno\b"), "No"),
            (re.compile(r"\bhas\s+personal\s+jurisdiction\b"), "Yes"),
            (re.compile(r"\bno\s+personal\s+jurisdiction\b"), "No"),
        ],
        instruction=(
            "Determine if the forum court could exercise personal jurisdiction over the defendant. "
            "Answer with exactly: 'Yes' or 'No'."
        ),
    ),
    "proa": TaskConfig(
        name="proa",
        labels=["Yes", "No"],
        normalizers=[
            (re.compile(r"\byes\b"), "Yes"),
            (re.compile(r"\bno\b"), "No"),
            (re.compile(r"\bprivate\s+right\s+of\s+action\b"), "Yes"),
            (re.compile(r"\bno\s+private\s+right\s+of\s+action\b"), "No"),
        ],
        instruction=(
            "Decide whether the statute text contains an explicit private right of action. "
            "Answer with exactly: 'Yes' or 'No'."
        ),
    ),
    "privacy_policy_entailment": TaskConfig(
        name="privacy_policy_entailment",
        labels=["Correct", "Incorrect"],
        normalizers=[
            (re.compile(r"\bcorrect\b"), "Correct"),
            (re.compile(r"\bincorrect\b"), "Incorrect"),
            (re.compile(r"\b(entails|supported)\b"), "Correct"),
            (re.compile(r"\b(contradicts|not supported|does not entail)\b"), "Incorrect"),
        ],
        instruction=(
            "Given a privacy policy clause and a description, decide if the description is correct. "
            "Answer with exactly: 'Correct' or 'Incorrect'."
        ),
    ),
    "insurance_policy_interpretation": TaskConfig(
        name="insurance_policy_interpretation",
        labels=["A", "B", "C"],
        normalizers=[
            (re.compile(r"\ba\b"), "A"),
            (re.compile(r"\bb\b"), "B"),
            (re.compile(r"\bc\b"), "C"),
            (re.compile(r"\byes\b"), "A"),
            (re.compile(r"\bno\b"), "B"),
            (re.compile(r"\bambig|can't decide|cannot decide\b"), "C"),
        ],
        instruction=(
            "Read the insurance policy and claim. Choose: "
            "[A: Yes (covered); B: No (not covered); C: It's ambiguous]. "
            "Answer with exactly one of: A, B, or C."
        ),
    ),
        "consumer_contracts_qa": TaskConfig(
        name="consumer_contracts_qa",
        labels=["Yes", "No"],
        normalizers=[
            (re.compile(r"^\s*yes\s*$"), "Yes"),
            (re.compile(r"^\s*no\s*$"), "No"),
            (re.compile(r"\byes\b"), "Yes"),
            (re.compile(r"\bno\b"), "No"),
        ],
        instruction=(
            "Answer the yes/no question about the consumer contract excerpt. "
            "Answer with exactly: 'Yes' or 'No'."
        ),
    ),

}
# -------------------------------
# Auto-add all other non-manual, auto-gradable LegalBench tasks
# -------------------------------

from datasets import get_dataset_config_names, load_dataset


def _infer_label_key(example: dict) -> Optional[str]:
    # LegalBench usually uses one of these keys
    for k in ("answer", "label", "output", "target", "gold"):
        if k in example:
            return k
    return None


def _looks_open_generation(label_values: List[str]) -> bool:
    """
    Heuristic filter for manual/open-generation tasks:
    - too many unique labels
    - labels are long / sentence-like
    """
    if not label_values:
        return True

    uniq = set(label_values)
    if len(uniq) > 30:
        return True

    avg_len = sum(len(x) for x in label_values) / max(1, len(label_values))
    if avg_len > 40:
        return True

    sentencey = 0
    for x in label_values:
        if len(x) >= 25 and (" " in x) and any(p in x for p in (".", ",", ";", ":")):
            sentencey += 1
    if sentencey / max(1, len(label_values)) > 0.3:
        return True

    return False


def _build_instruction_and_normalizers(labels: List[str]) -> Tuple[str, List[Tuple[Pattern[str], str]]]:
    """
    Create label-only instructions and simple regex normalizers consistent with your earlier style.
    """
    # Canonical special-cases
    lower_set = {x.lower() for x in labels}

    # Yes/No
    if lower_set == {"yes", "no"}:
        instr = "Answer with exactly: 'Yes' or 'No'."
        norms = [
            (re.compile(r"^\s*yes\s*$"), "Yes"),
            (re.compile(r"^\s*no\s*$"), "No"),
            (re.compile(r"\byes\b"), "Yes"),
            (re.compile(r"\bno\b"), "No"),
        ]
        return instr, norms

    # Correct/Incorrect
    if lower_set == {"correct", "incorrect"}:
        instr = "Answer with exactly: 'Correct' or 'Incorrect'."
        norms = [
            (re.compile(r"^\s*correct\s*$"), "Correct"),
            (re.compile(r"^\s*incorrect\s*$"), "Incorrect"),
            (re.compile(r"\bcorrect\b"), "Correct"),
            (re.compile(r"\bincorrect\b"), "Incorrect"),
        ]
        return instr, norms

    # True/False
    if lower_set == {"true", "false"}:
        instr = "Answer with exactly: 'True' or 'False'."
        norms = [
            (re.compile(r"^\s*true\s*$"), "True"),
            (re.compile(r"^\s*false\s*$"), "False"),
            (re.compile(r"\btrue\b"), "True"),
            (re.compile(r"\bfalse\b"), "False"),
        ]
        return instr, norms

    # A/B/C (common multiple-choice)
    if set(labels) == {"A", "B", "C"}:
        instr = "Answer with exactly one of: A, B, or C."
        norms = [
            (re.compile(r"^\s*a\s*$"), "A"),
            (re.compile(r"^\s*b\s*$"), "B"),
            (re.compile(r"^\s*c\s*$"), "C"),
            (re.compile(r"\ba\b"), "A"),
            (re.compile(r"\bb\b"), "B"),
            (re.compile(r"\bc\b"), "C"),
        ]
        return instr, norms

    # Generic label-only instruction
    joined = ", ".join(labels)
    instr = f"Answer with exactly one of: {joined}."

    # Build normalizers: allow exact label match, or case-insensitive match of the label
    norms: List[Tuple[Pattern[str], str]] = []
    for lab in labels:
        lab_esc = re.escape(lab.lower())
        norms.append((re.compile(rf"^\s*{lab_esc}\s*$"), lab))
        norms.append((re.compile(rf"\b{lab_esc}\b"), lab))

    return instr, norms


def add_all_non_manual_tasks_to_TASKS(
    *,
    dataset_id: str = "nguha/legalbench",
    trust_remote_code: bool = True,
    max_probe: int = 200,
) -> None:
    """
    Adds all LegalBench configs that look auto-gradable (discrete label sets)
    to TASKS, without touching tasks you already defined manually above.
    """
    config_names = get_dataset_config_names(dataset_id)

    # Explicitly exclude known manual/open-gen anchor
    explicit_exclude = {"rule_qa"}

    for cfg in config_names:
        if cfg in TASKS:
            continue
        if cfg in explicit_exclude:
            continue

        try:
            ds = load_dataset(dataset_id, cfg, trust_remote_code=trust_remote_code)
        except Exception:
            # Skip configs that can't be loaded in this environment
            continue

        if "test" not in ds or len(ds["test"]) == 0:
            continue

        ex0 = ds["test"][0]
        label_key = _infer_label_key(ex0)
        if not label_key:
            continue

        # Probe label values from train/test
        label_values: List[str] = []
        for split in ("train", "test"):
            if split not in ds:
                continue
            for i, ex in enumerate(ds[split]):
                if i >= max_probe:
                    break
                v = ex.get(label_key)
                if v is None:
                    continue
                label_values.append(str(v))

        if _looks_open_generation(label_values):
            # Likely rule-application or other manual/open-gen tasks
            continue

        labels = sorted(set(label_values))
        instruction_suffix, normalizers = _build_instruction_and_normalizers(labels)

        TASKS[cfg] = TaskConfig(
            name=cfg,
            labels=labels,
            normalizers=normalizers,
            instruction=(
                "Follow the task instruction exactly. Output only the label; no explanation. "
                + instruction_suffix
            ),
        )


# Auto-populate on import so run_eval.py can use any auto-gradable task name.
add_all_non_manual_tasks_to_TASKS()

