import pytest

from tools.run_hardware_scene import run_scene


@pytest.mark.usefixtures("local_hardware_examples")
def test_actual_bucky_binary_prefix_through_driver_matches_independent_rasterizer(tmp_path):
    report = run_scene("bucky12", tmp_path)
    assert report["fence"] == "completed"
    assert report["host_instructions"] == 1973855
    assert report["color_mismatches"] == 0
    assert report["nonzero_support_mismatches"] == 0
    assert not report["actual_fpga_execution"]
    assert (tmp_path / "frame.png").is_file()


def test_utah_rounding_approximation_cannot_be_enabled_implicitly(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        run_scene("utah", tmp_path)


@pytest.mark.usefixtures("local_hardware_examples")
def test_managed_utah_binary_reports_real_approximation_difference(tmp_path):
    report = run_scene("utah", tmp_path, allow_approximate_fp16=True)
    assert report["fence"] == "completed"
    assert report["entry_pc"] == 123
    assert report["submitted_prefix_words"] == 496
    assert report["host_instructions"] == 2188023
    assert report["color_mismatches"] == 75
    assert report["nonzero_support_mismatches"] == 0
    assert not report["pixel_exact_reference_match"]
    assert not report["actual_fpga_execution"]
