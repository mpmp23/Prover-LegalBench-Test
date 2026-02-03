# LegalBench + DeepSeek-Prover-V2 (OpenRouter) starter benchmarks

This is a minimal evaluation test for a recommended subset of LegalBench tasks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Setup environment variables (choose one method):

# Method 1: Create a .env file (recommended)
cp env.example .env
# Then edit .env and add your API key

# Method 2: Export in your shell
export OPENROUTER_API_KEY="your-key-here"
export HF_TOKEN="your-hf-token-here"  # Optional
export OPENROUTER_HTTP_REFERER="http://localhost:8000"  # Optional
export OPENROUTER_X_TITLE="LegalBench-Eval"  # Optional
```

## Run

```bash
# Test with DeepSeek R1 model
python run_eval.py --model "deepseek/deepseek-r1" --tasks hearsay --n_shots 0 --max_test 1

# Run full evaluation with DeepSeek Prover V2
python run_eval.py --model "deepseek-ai/deepseek-prover-v2-671b" --n_shots 3 --max_test 100

# Run with other models
python run_eval.py --model "openai/gpt-4" --tasks hearsay personal_jurisdiction --n_shots 3 --max_test 50
```

Results are written to `runs/<task>.jsonl` and `runs/summary.json`.

And, I'd add in run_eval.py --discover_all once to load in all of your tests ahead of that aren't teh five manually added to task_configs.

## Model Support

The client automatically handles provider routing:
- **DeepSeek Prover models** (`deepseek-prover`): Forces `novita`/`azure` providers
- **Other models** (DeepSeek R1, GPT, Claude, etc.): Lets OpenRouter choose the best provider

## Add more tasks

1. Add the task name to `TASKS` in `task_configs.py`.
2. Specify the canonical labels and normalization rules (regexes).
3. Run: `python run_eval.py --tasks your_task_name`.
