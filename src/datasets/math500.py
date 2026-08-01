from datasets import load_dataset
import random

DATASET_NAME = "HuggingFaceH4/MATH-500"
SEED = 42
N_SAMPLES = 500

# ---------------------------------------------------------------------------
# Fixed 100-sample subset
# ---------------------------------------------------------------------------

def load_fixed_subset(n: int = N_SAMPLES, seed: int = SEED):
    ds = load_dataset(DATASET_NAME, split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return ds.select(idx[:n])

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )