"""Tests for the SMTP send retry in EmailClient (transient DNS/network blips)."""

import socket
from unittest.mock import patch

from orchestrator.services.gmail_client import EmailClient


class _FlakySMTP:
    """Raises socket.gaierror on the first two connects, then succeeds."""

    calls = 0

    def __init__(self, host, port, timeout=30):
        _FlakySMTP.calls += 1
        if _FlakySMTP.calls < 3:
            raise socket.gaierror(-2, "Name or service not known")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        pass

    def login(self, email, password):
        pass

    def send_message(self, msg):
        pass


class _NeverSMTP:
    """Always fails DNS; retries must exhaust and re-raise."""

    def __init__(self, host, port, timeout=30):
        raise socket.gaierror(-2, "Name or service not known")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _client():
    return EmailClient(
        email_address="sender@example.com",
        password="pw",
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        imap_host="imap.gmail.com",
        imap_port=993,
    )


def test_send_message_retries_transient_dns_failure():
    _FlakySMTP.calls = 0
    with patch("orchestrator.services.gmail_client.smtplib.SMTP", _FlakySMTP):
        result = _client().send_message("to@example.com", "Subject", "Body")
    assert result["status"] == "sent"
    assert _FlakySMTP.calls == 3


def test_send_message_exhausts_retries_then_raises():
    with patch("orchestrator.services.gmail_client.smtplib.SMTP", _NeverSMTP):
        try:
            _client().send_message("to@example.com", "Subject", "Body")
        except socket.gaierror:
            pass
        else:
            raise AssertionError("expected socket.gaierror to propagate")
