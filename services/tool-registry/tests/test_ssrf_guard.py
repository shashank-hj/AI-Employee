import pytest
from tool_registry.services.ssrf_guard import SSRFError, validate_http_url


class TestSSRFGuard:
    def test_accepts_public_https(self):
        validate_http_url("https://example.com/api")  # should not raise

    def test_rejects_non_http_scheme(self):
        with pytest.raises(SSRFError):
            validate_http_url("file:///etc/passwd")
        with pytest.raises(SSRFError):
            validate_http_url("ftp://example.com")

    def test_rejects_localhost(self):
        with pytest.raises(SSRFError):
            validate_http_url("http://localhost:8080/x")
        with pytest.raises(SSRFError):
            validate_http_url("http://127.0.0.1/x")

    def test_rejects_private_ranges(self):
        for url in (
            "http://10.0.0.5/x",
            "http://192.168.1.1/x",
            "http://172.16.5.5/x",
            "http://169.254.169.254/latest/meta-data",
            "http://0.0.0.0/x",
        ):
            with pytest.raises(SSRFError):
                validate_http_url(url)

    def test_rejects_private_hostnames(self):
        for url in (
            "http://host.docker.internal:11434",
            "http://metadata.google.internal",
        ):
            with pytest.raises(SSRFError):
                validate_http_url(url)

    def test_rejects_local_suffix(self):
        with pytest.raises(SSRFError):
            validate_http_url("http://server.local/x")

    def test_rejects_no_host(self):
        with pytest.raises(SSRFError):
            validate_http_url("https:///path")
