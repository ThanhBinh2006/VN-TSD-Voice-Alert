import pytest

from ts_alert.runtime.batch import per_rank_batch


def test_per_rank_batch_splits_total():
    assert per_rank_batch(8, world_size=2) == 4
    assert per_rank_batch(8, world_size=1) == 8


def test_per_rank_batch_not_divisible():
    with pytest.raises(ValueError):
        per_rank_batch(7, world_size=2)


def test_per_rank_batch_reads_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert per_rank_batch(8) == 4
