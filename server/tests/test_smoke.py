import svd_server


def test_package_exposes_version() -> None:
    assert svd_server.__version__ == "0.0.0"
