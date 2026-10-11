import pytest


@pytest.mark.skip(reason="TODO: Thành viên 4 hiện thực test này")
def test_dedup():
    """Deduplicator: cùng biển xuất hiện 30 khung liên tiếp chỉ sinh 1 SpeechEvent; vắng > lost_frames rồi xuất hiện lại -> sự kiện mới"""
