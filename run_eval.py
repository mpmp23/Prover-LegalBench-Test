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

    # Initialize the client
    client = OpenAICompatibleChatClient(
        model=args.model,
        http_referer=args.http_referer,
        x_title=args.x_title
    )

    summary: Dict[str, Any] = {"model": args.model, "tasks": {}}

    for task_name in args.tasks:
        if task_name not in TASKS:
            print(f"Skipping '{task_name}': not in TASKS config.")
            continue
        
        task = TASKS[task_name]

        # 1. Load the dataset for this specific task
        print(f"[{task_name}] Loading dataset...")
        ds = load_dataset("nguha/legalbench", task_name, trust_remote_code=True)
        train = list(ds["train"])
        test = list(ds["test"])

        if not test:
            print(f"[{task_name}] No test split found; skipping.")
            continue

        # 2. Reset counters and setup paths for this task
        correct = 0
        total = 0
        rows: List[Dict[str, Any]] = []
        out_path = os.path.join(
            args.out_dir,
            f"{task_name}_shots{args.n_shots}_seed{args.seed}_max{args.max_test}.jsonl"
        )

        # Clear file if it exists to avoid appending to old runs
        if os.path.exists(out_path):
            os.remove(out_path)

        label_key = infer_label_key(test[0])
        if not label_key:
            print(f"[{task_name}] Could not infer label key; skipping.")
            continue

        fewshot = pick_fewshot(train, args.n_shots, args.seed, label_key=label_key)

        # 3. Evaluation Loop
        print(f"[{task_name}] Starting eval (max={args.max_test})...")
        for i, ex in enumerate(test[: args.max_test]):
            gold = ex[label_key]
            prompt = build_prompt(task, ex, fewshot, label_key=label_key)

            messages = [
                {"role": "system", "content": "Follow the task instruction exactly. Output only the label; no explanation."},
                {"role": "user", "content": prompt},
            ]

            try:
                # Attempt API call
                raw = client.complete(messages, temperature=0.0)
            except Exception as e:
                print(f"[{task_name}] Error on example {i}: {e}")
                raw = f"ERROR_API_CALL: {str(e)}"

            # Process the prediction
            pred = task.normalize(raw)
            is_correct = (pred == gold) if pred is not None else False
            
            correct += int(is_correct)
            total += 1

            # Prepare row data
            row = {
                "task": task_name,
                "i": i,
                "gold": gold,
                "raw": raw,
                "pred": pred,
                "correct": is_correct,
            }
            rows.append(row)

            # Write incrementally to JSONL
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                print(f"[{task_name}] {i+1}/{min(len(test), args.max_test)} processed...")

        # 4. Finalize Task Stats
        acc = correct / total if total else 0.0
        summary["tasks"][task_name] = {
            "n": total,
            "accuracy": acc,
        }
        print(f"[{task_name}] Completed. Accuracy: {acc:.3f}")

    # Write the final summary
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation finished. Summary written to {summary_path}")


if __name__ == "__main__":
    main()