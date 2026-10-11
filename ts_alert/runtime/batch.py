"""Tiện ích chạy đa GPU (DDP với torchrun): chia batch tổng thành batch cho từng GPU (rank)."""
from __future__ import annotations

import os


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    """Chỉ rank 0 mới nên ghi log/checkpoint/metrics."""
    return get_rank() == 0


def per_rank_batch(total_batch: int, world_size: int | None = None) -> int:
    """train.batch_size trong shared.yaml là TỔNG trên mọi GPU -> mỗi GPU dùng total // world_size."""
    ws = world_size or get_world_size()
    if total_batch % ws != 0:
        raise ValueError(f"batch_size={total_batch} không chia hết cho số GPU={ws}")
    return total_batch // ws
