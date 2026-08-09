from datasets import load_dataset
import random

SEED = 42

# ---------------------------------------------------------------------------
# GPQA dataset loading & MCQ formatting
# ---------------------------------------------------------------------------

def load_gpqa():
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    return ds


def format_mcq(ex, idx: int):
    """Shuffle answer choices deterministically per question idx."""
    correct  = ex["Correct Answer"]
    wrongs   = [ex["Incorrect Answer 1"],
                ex["Incorrect Answer 2"],
                ex["Incorrect Answer 3"]]
    choices  = [correct] + wrongs
    rng      = random.Random(SEED + idx)
    rng.shuffle(choices)
    labels   = "ABCD"
    correct_label = labels[choices.index(correct)]
    choice_text   = "\n".join(f"({labels[i]}) {choices[i]}" for i in range(4))
    prompt_body = (
        f"{ex['Question']}\n\n"
        f"{choice_text}\n\n"
        "Please reason step by step and select the single best answer. "
        "At the end of your response, write exactly: "
        "'The correct answer is (X).' where X is A, B, C, or D."
    )
    return prompt_body, correct_label


def build_prompt(tokenizer, question_body: str, enable_thinking: bool = True) -> str:
    msgs = [{"role": "user", "content": question_body}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(msgs, **kwargs)
