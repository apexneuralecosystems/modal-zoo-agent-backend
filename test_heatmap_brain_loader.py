import heatmap_brain_loader as L
import heatmap_brain_default as default


def test_no_url_uses_default():
    assert L.load_brain({}, download_brain=lambda u: "x.py") is default


def test_download_error_falls_back_to_default():
    def boom(_u):
        raise RuntimeError("network down")
    job = {"brain_presigned_url": "http://x", "brain_version": 3, "camera_id": "c"}
    assert L.load_brain(job, download_brain=boom) is default


def test_valid_brain_is_loaded(tmp_path):
    p = tmp_path / "brain.py"
    p.write_text("def setup(ctx):\n    pass\ndef heat_mask(f, t):\n    return None\n")
    job = {"brain_presigned_url": "http://x", "brain_version": 5, "camera_id": "c"}
    mod = L.load_brain(job, download_brain=lambda _u: str(p))
    assert mod is not default
    assert hasattr(mod, "setup") and hasattr(mod, "heat_mask")


def test_brain_missing_interface_falls_back(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("x = 1\n")  # no setup/heat_mask
    job = {"brain_presigned_url": "http://x", "brain_version": 6, "camera_id": "c"}
    assert L.load_brain(job, download_brain=lambda _u: str(p)) is default
