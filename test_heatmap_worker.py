from heatmap_worker import local_day, due_to_upload


def test_local_day_formats_yyyy_mm_dd():
    d = local_day(1_781_000_000.0)
    assert len(d) == 10
    assert d[4] == '-' and d[7] == '-'


def test_local_day_changes_across_a_day_boundary():
    # two timestamps ~2 days apart must produce different day strings
    a = local_day(1_781_000_000.0)
    b = local_day(1_781_000_000.0 + 2 * 24 * 3600)
    assert a != b


def test_due_to_upload_true_at_or_after_interval():
    assert due_to_upload(1000.0, 1000.0 + 30 * 60, 30) is True
    assert due_to_upload(1000.0, 1000.0 + 31 * 60, 30) is True


def test_due_to_upload_false_before_interval():
    assert due_to_upload(1000.0, 1000.0 + 29 * 60, 30) is False
