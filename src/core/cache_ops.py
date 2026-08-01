import torch
from transformers import DynamicCache

# ============================================================
# Parameters
# ============================================================
BLOCK_SIZE   = 16   # PagedAttention 정렬 단위
RECENT_SIZE  = 16   # recent window 보호 (뒤 N개)

# ============================================================
# BlockRegistry: RoPE position 추적
# ============================================================
class BlockRegistry:
    """
    KV eviction 후에도 각 토큰의 절대 position을 추적.

    핵심 공식:
        절대 position = block_id × BLOCK_SIZE + local_idx

    eviction 후 Q의 position_ids를 올바르게 override하기 위해
    살아있는 토큰들의 절대 position 목록을 항상 유지.
    """

    def __init__(self):
        # {block_id: {"start_pos": int, "mask": list[bool]}}
        self.blocks = {}
        self.next_abs_pos = 0

    def add_tokens_batch(self, count: int, start_pos: int = 0):
        for i in range(count):
            self._add_one(start_pos + i)
        self.next_abs_pos = start_pos + count

    def _add_one(self, abs_pos: int):
        block_id  = abs_pos // BLOCK_SIZE
        local_idx = abs_pos %  BLOCK_SIZE
        if block_id not in self.blocks:
            self.blocks[block_id] = {
                "start_pos": block_id * BLOCK_SIZE,
                "mask": [False] * BLOCK_SIZE
            }
        self.blocks[block_id]["mask"][local_idx] = True

    def evict_by_cache_idx(self, cache_idx: int):
        """cache 내 순서 인덱스 → 절대 position으로 변환 후 evict"""
        alive = self.get_alive_positions()
        if cache_idx >= len(alive):
            return
        abs_pos   = alive[cache_idx]
        block_id  = abs_pos // BLOCK_SIZE
        local_idx = abs_pos %  BLOCK_SIZE
        self.blocks[block_id]["mask"][local_idx] = False

    def get_alive_positions(self) -> list:
        alive = []
        for bid in sorted(self.blocks.keys()):
            b = self.blocks[bid]
            for li, is_alive in enumerate(b["mask"]):
                if is_alive:
                    alive.append(b["start_pos"] + li)
        return alive

    def get_next_position_id(self) -> int:
        """다음 생성 토큰의 절대 position (HuggingFace position_ids override용)"""
        return self.next_abs_pos

    def register_new_token(self):
        """decoding step마다 새 토큰 등록"""
        self._add_one(self.next_abs_pos)
        self.next_abs_pos += 1

    def alive_count(self)  -> int: return len(self.get_alive_positions())
    def total_seen(self)   -> int: return self.next_abs_pos
    def eviction_rate(self)-> float:
        t = self.total_seen()
        return 0.0 if t == 0 else 1.0 - self.alive_count() / t

    def __repr__(self):
        return (f"BlockRegistry(total={self.total_seen()}, "
                f"alive={self.alive_count()}, "
                f"rate={self.eviction_rate():.1%})")
    
    def evict_by_cache_indices(self, cache_indices: list):
        """여러 cache_idx를 한 번에 절대 position으로 변환 후 evict (O(N) 병목 해결)"""
        # alive 목록을 딱 한 번만 만듦!
        alive = self.get_alive_positions()
        
        for idx in cache_indices:
            if idx < len(alive):
                abs_pos = alive[idx]
                block_id  = abs_pos // BLOCK_SIZE
                local_idx = abs_pos %  BLOCK_SIZE
                self.blocks[block_id]["mask"][local_idx] = False



# ============================================================
# KV Cache 실제 eviction 유틸 (transformers 5.x 호환)
# ============================================================
def evict_from_cache(cache: DynamicCache, keep_indices: list) -> DynamicCache:
    """Remove all positions NOT in keep_indices from a DynamicCache."""
    keep = torch.tensor(keep_indices, dtype=torch.long, device="cuda")
    if hasattr(cache, "key_cache"):
        for li in range(len(cache.key_cache)):
            k = cache.key_cache[li]
            v = cache.value_cache[li]
            if k is not None and k.numel() > 0:
                cache.key_cache[li]   = k[:, :, keep, :]
                cache.value_cache[li] = v[:, :, keep, :]
    elif hasattr(cache, "layers") and len(cache.layers) > 0:
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is not None and k.numel() > 0:
                layer.keys   = k[:, :, keep, :]
                layer.values = v[:, :, keep, :]
    return cache


def cache_memory_bytes(cache: DynamicCache) -> int:
    total = 0
    for layer in cache.layers:
        total += layer.keys.numel()   * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total


def register_v_hook(model):
    v_storage = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inp, output):
            v_storage[layer_idx] = output.detach()
        return hook

    for i, layer in enumerate(model.model.layers):
        li = i + 1
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(li)))
    return handles, v_storage


def remove_hooks(handles):
    for h in handles:
        h.remove()

# ---------------------------------------------------------------------------
# K-norm from KV cache
# ---------------------------------------------------------------------------

def extract_last_position_knorm(cache) -> float:
    layer_scores = []
    if hasattr(cache, "key_cache"):
        for layer_keys in cache.key_cache:
            if layer_keys is None or layer_keys.numel() == 0:
                continue
            last = layer_keys[:, :, -1, :]
            layer_scores.append(last.float().norm(dim=-1).mean(dim=(0, 1)))
    elif hasattr(cache, "layers") and len(cache.layers) > 0:
        for layer in cache.layers:
            layer_keys = getattr(layer, "keys", None)
            if layer_keys is None or layer_keys.numel() == 0:
                continue
            last = layer_keys[:, :, -1, :]
            layer_scores.append(last.float().norm(dim=-1).mean(dim=(0, 1)))
    else:
        raise RuntimeError(
            "Could not find K vectors in cache — unrecognized DynamicCache structure."
        )
    return torch.stack(layer_scores).mean().item()