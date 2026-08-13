"""
AIME 2024 I+II / 2025 I+II — 60문제 로더
=========================================
Reference:
  - https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
  - https://huggingface.co/datasets/opencompass/AIME2025
"""

from pathlib import Path
from datasets import load_dataset, concatenate_datasets

DEFAULT_PATH = Path("data/aime_2024_2025.jsonl")


def load_aime() -> list[dict]:
    ds_2024    = load_dataset("HuggingFaceH4/aime_2024",  split="train")
    ds_2025_I  = load_dataset("opencompass/AIME2025", "AIME2025-I",  split="test")
    ds_2025_II = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")

    problems = []
    for ex in ds_2024:                          # 컬럼: problem / answer
        problems.append({
            "problem": ex["problem"],
            "answer":  int(ex["answer"]),
        })
    for ds in [ds_2025_I, ds_2025_II]:         # 컬럼: question / answer
        for ex in ds:
            problems.append({
                "problem": ex["question"],
                "answer":  int(ex["answer"]),
            })

    assert len(problems) == 60, f"Expected 60, got {len(problems)}"
    return problems

def build_prompt(tokenizer, question: str, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": question}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(messages, **kwargs)