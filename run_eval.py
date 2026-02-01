import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

from datasets import load_dataset

from openrouter_client import OpenAICompatibleChatClient
from task_configs import TASKS
from eval_utils import build_prompt, pick_fewshot, infer_label_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                    help="LegalBench task names (HF config names). Default: recommended set.")
    ap.add_argument("--n_shots", type=int, default=3, help="Few-shot examples sampled from train split.")
    ap.add_argument("--max_test", type=int, default=100, help="Max test examples per task.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-Prover-V2-671B:novita")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--http_referer", default=os.environ.get("OPENROUTER_HTTP_REFERER"))
    ap.add_argument("--x_title", default=os.environ.get("OPENROUTER_X_TITLE", "legalbench-eval"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    client = OpenAICompatibleChatClient(
        model=args.model,
    )


    summary: Dict[str, Any] = {"model": args.model, "tasks": {}}

    for task_name in args.tasks:
        if task_name not in TASKS:
            raise ValueError(f"Task '{task_name}' not in TASKS config. Add it in task_configs.py.")
        task = TASKS[task_name]

        ds = load_dataset("nguha/legalbench", task_name, trust_remote_code=True)

        train = list(ds["train"])
        test = list(ds["test"])

        if not test:
            print(f"[{task_name}] No test split found; skipping.")
            continue

        label_key = infer_label_key(test[0])
        if not label_key:
            raise RuntimeError(f"[{task_name}] Could not infer label key from example keys: {list(test[0].keys())}")

        fewshot = pick_fewshot(train, args.n_shots, args.seed, label_key=label_key)

        correct = 0
        total = 0
        rows: List[Dict[str, Any]] = []

        for i, ex in enumerate(test[: args.max_test]):
            gold = ex[label_key]
            prompt = build_prompt(task, ex, fewshot, label_key=label_key)

            messages = [
                {"role": "system", "content": "Follow the task instruction exactly. Output only the label; no explanation."},
                {"role": "user", "content": prompt},
            ]
            raw = client.complete(messages, temperature=0.0)
            pred = task.normalize(raw)

            is_correct = (pred == gold) if pred is not None else False
            correct += int(is_correct)
            total += 1

            rows.append({
                "task": task_name,
                "i": i,
                "gold": gold,
                "raw": raw,
                "pred": pred,
                "correct": is_correct,
            })

            if (i + 1) % 20 == 0:
                print(f"[{task_name}] {i+1}/{min(len(test), args.max_test)} done...")

        acc = correct / total if total else 0.0
        summary["tasks"][task_name] = {
            "n": total,
            "accuracy": acc,
        }

        out_path = os.path.join(
        args.out_dir,
        f"{task_name}_shots{args.n_shots}_seed{args.seed}_max{args.max_test}.jsonl"
    )
        
        # Write per-example results
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[{task_name}] accuracy={acc:.3f} (n={total}) -> {out_path}")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Wrote", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
