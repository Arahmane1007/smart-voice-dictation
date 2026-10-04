import svd_client_windows


def test_package_exposes_version() -> None:
    assert svd_client_windows.__version__ == "0.0.0"
