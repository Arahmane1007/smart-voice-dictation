import svd_core


def test_package_exposes_version() -> None:
    assert svd_core.__version__ == "0.0.0"
