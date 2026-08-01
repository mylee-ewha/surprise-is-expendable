import torch
import numpy as np
from transformers import DynamicCache

from .cache_ops import remove_hooks, register_v_hook, cache_memory_bytes, BlockRegistry, extract_last_position_knorm, RECENT_SIZE, evict_from_cache
from .scorers import PerSampleScorer, N_LAYERS
from ..utils.metrics import extract_boxed_answer


EVICT_INTERVAL = 128  # lazy eviction 간격
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
GEN_SEED = 42
MAX_NEW_TOKENS = 8192

_METHOD_NEEDS = {
    "baseline":   {"hidden_states": False, "v_hook": False},
    "k_norm":     {"hidden_states": False, "v_hook": False},
    "donut_a_v2": {"hidden_states": True,  "v_hook": False},
    "novelty":    {"hidden_states": False, "v_hook": True},
    "donut_a_v2_inv": {"hidden_states": True,  "v_hook": False}, 
    "novelty_inv":    {"hidden_states": False, "v_hook": True}, 
    "v_angular":     {"hidden_states": False, "v_hook": True},
    "v_angular_inv": {"hidden_states": False, "v_hook": True},
}

_EVICT_HIGHEST = {"donut_a_v2_inv", "novelty_inv", "v_angular_inv"}

def sample_next_token(logits, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K):
    logits = logits.float() / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, k, dim=-1)[0][..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# ---------------------------------------------------------------------------
# Generation with scored streaming eviction (thinking-block only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_scored_eviction(
    model, tokenizer, prompt, method, budget, device,
    max_new_tokens=MAX_NEW_TOKENS,
):
    # ── [개선 2] method별로 필요한 부가 출력만 활성화 ─────────────────────
    needs_hidden = _METHOD_NEEDS[method]["hidden_states"]
    needs_v_hook = _METHOD_NEEDS[method]["v_hook"]

    if method == "baseline":
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        torch.manual_seed(GEN_SEED)
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        truncated = "</think>" not in text
        if not truncated:
            think_tokens_generated = len(
                tokenizer.encode(text.split("</think>")[0], add_special_tokens=False)
            )
        else:
            think_tokens_generated = len(gen_ids)
        return {
            "text": text,
            "pred": extract_boxed_answer(text),
            "think_tokens_generated": think_tokens_generated,
            "total_tokens_generated": len(gen_ids),
            "truncated": truncated,
        }

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    # ── v_hook: novelty일 때만 등록 ───────────────────────────────────────
    if needs_v_hook:
        handles, v_storage = register_v_hook(model)
    else:
        handles, v_storage = [], {}

    try:
        prefill_out = model(
            input_ids, use_cache=True,
            output_hidden_states=needs_hidden,
            past_key_values=DynamicCache(),
        )
        cache = prefill_out.past_key_values
        next_logits = prefill_out.logits[:, -1, :]
        memory_before = cache_memory_bytes(cache)

        reg = BlockRegistry()
        reg.add_tokens_batch(prompt_len)

        scorer = PerSampleScorer(method, device)
        think_scores = []
        think_generated_count = 0
        n_evicted = 0
        in_think = True

        generated_ids = []
        generated_text_buf = ""
        torch.manual_seed(GEN_SEED)
        next_token = sample_next_token(next_logits)
        generated_ids.append(next_token.item())

        for _ in range(max_new_tokens):
            if next_token.item() == tokenizer.eos_token_id:
                break

            pos_id = torch.tensor(
                [[reg.get_next_position_id()]], dtype=torch.long, device=device
            )
            # ── [개선 2] decode step: method별 필요한 출력만 요청 ─────────
            step_out = model(
                next_token, use_cache=True,
                output_hidden_states=needs_hidden,
                past_key_values=cache, position_ids=pos_id,
            )
            cache = step_out.past_key_values
            next_logits = step_out.logits[:, -1, :]
            reg.register_new_token()

            if in_think:
                if method == "k_norm":
                    score = extract_last_position_knorm(cache)
                elif method in ("donut_a_v2", "donut_a_v2_inv"): 
                    hs = [step_out.hidden_states[i][0, 0] for i in range(0, N_LAYERS + 1)]
                    score = scorer.score_donut_a_v2(hs)
                elif method in ("novelty", "novelty_inv"):
                    score = scorer.score_novelty(v_storage)
                else:  # v_angular, v_angular_inv
                    score = scorer.score_v_angular(v_storage)
                think_scores.append(score)
                think_generated_count += 1

            piece = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            generated_text_buf += piece
            if in_think and "</think>" in generated_text_buf:
                in_think = False

            should_evict = (
                len(think_scores) > budget and (
                    think_generated_count % EVICT_INTERVAL == 0 or
                    len(think_scores) >= budget + EVICT_INTERVAL
                )
            )
            if should_evict:
                n_to_evict = len(think_scores) - budget
                n_cand     = len(think_scores) - RECENT_SIZE
                if n_cand > 0:
                    valid = [(j, s) for j, s in enumerate(think_scores[:n_cand])
                            if not np.isnan(s)]
                    if valid:
                        n_actual = min(n_to_evict, len(valid))
                        sorted_valid = sorted(valid, key=lambda p: p[1],
                                            reverse=(method in _EVICT_HIGHEST))
                        evict_js = sorted(
                            [j for j, _ in sorted_valid[:n_actual]], reverse=True
                        )
                        evict_set = {prompt_len + j for j in evict_js}
                        n_alive   = cache.get_seq_length()
                        keep_list = [i for i in range(n_alive) if i not in evict_set]
                        evict_from_cache(cache, keep_list)          # 텐서 재구성 1회
                        reg.evict_by_cache_indices([prompt_len + j for j in evict_js])
                        for j in evict_js:                          # 역순이라 인덱스 안전
                            del think_scores[j]
                        n_evicted += len(evict_js)

            next_token = sample_next_token(next_logits)
            generated_ids.append(next_token.item())
    finally:
        if needs_v_hook:
            remove_hooks(handles)

    memory_after = cache_memory_bytes(cache)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "text": text,
        "pred": extract_boxed_answer(text),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "budget": budget,
        "think_tokens_generated": think_generated_count,
        "total_tokens_generated": len(generated_ids),
        "truncated": in_think,
        "n_evicted": n_evicted,
        "final_seq_len": cache.get_seq_length(),
    }
