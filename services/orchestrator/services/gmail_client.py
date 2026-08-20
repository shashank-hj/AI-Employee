"""Email client for reading and sending emails via SMTP + IMAP (e.g. Gmail)."""

import email
import imaplib
import re
import smtplib
import socket
import time
from email.header import decode_header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import structlog

from orchestrator.config import settings

logger = structlog.get_logger(__name__)


class EmailClient:
    def __init__(
        self,
        email_address: str | None = None,
        password: str | None = None,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        imap_host: str | None = None,
        imap_port: int | None = None,
        display_name: str | None = None,
    ) -> None:
        self._email = email_address or settings.EMAIL_ADDRESS
        self._password = password or settings.EMAIL_PASSWORD
        self._smtp_host = smtp_host or settings.EMAIL_SMTP_HOST
        self._smtp_port = smtp_port or settings.EMAIL_SMTP_PORT
        self._imap_host = imap_host or settings.EMAIL_IMAP_HOST
        self._imap_port = imap_port or settings.EMAIL_IMAP_PORT
        self._display_name = display_name or settings.EMAIL_DISPLAY_NAME

    @property
    def enabled(self) -> bool:
        return bool(self._email and self._password)

    # ── SMTP: Send ──

    _TRANSIENT_SMTP_ERRORS = (
        socket.gaierror,
        OSError,
        smtplib.SMTPServerDisconnected,
    )

    def _is_transient_smtp_error(self, exc: Exception) -> bool:
        """Return True for network blips and transient Gmail protocol desyncs.

        Gmail intermittently replies ``555 5.5.2 Syntax error, cannot decode
        response`` on an otherwise valid session (a brief protocol desync, not
        a data problem). It self-heals on retry with a fresh connection. Real
        auth (``535``) and permanent rejections (``550/554``) are NOT retried.
        """
        if isinstance(exc, self._TRANSIENT_SMTP_ERRORS):
            return True
        if isinstance(exc, smtplib.SMTPResponseException):
            code = exc.smtp_code
            # 5xx that are genuinely transient/desync related:
            if code in (555, 421, 450, 451, 452):
                return True
        return False

    _EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    @staticmethod
    def _normalize_address(value: str | None, field: str = "recipient") -> str:
        """Extract one valid RFC 5321 address from free text.

        LLM-generated tool calls occasionally pass the raw user phrase as the
        recipient (e.g. ``"send to a@example.com"``). Put verbatim into the
        ``To`` header, smtplib copies it straight into the envelope ``RCPT TO``
        command, and the embedded spaces make Gmail reply with
        ``555 5.5.2 Syntax error``. Pull out the first well-formed address and
        fail fast with a clear message when none is present.
        """
        if not value or not str(value).strip():
            raise ValueError(f"Missing {field} email address")
        raw = str(value).strip()
        match = EmailClient._EMAIL_RE.search(raw)
        if not match:
            raise ValueError(
                f"Invalid {field} email address: {raw!r}. "
                "Expected a valid address like user@example.com"
            )
        return match.group(0)

    @staticmethod
    def _normalize_recipients(value: str | None) -> str | None:
        """Normalize a possibly comma-separated list of addresses."""
        if not value:
            return None
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            raise ValueError("Invalid CC email address")
        return ", ".join(EmailClient._normalize_address(p, "CC") for p in parts)

    def _smtp_send(self, msg: Any) -> None:
        """Send ``msg`` via SMTP, retrying transient network/DNS failures.

        ``[Errno -2] Name or service not known`` is a DNS resolution blip (the
        Docker resolver briefly failing to resolve smtp.gmail.com); retrying a
        couple of times resolves it without surfacing to the user. Auth errors
        are NOT retried.
        """
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=30) as server:
                    server.starttls()
                    server.login(self._email, self._password)
                    server.send_message(msg)
                return
            except Exception as exc:
                if not self._is_transient_smtp_error(exc):
                    raise
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        if last_error is not None:
            raise last_error

    def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        attachments: list[tuple[str, bytes, str]] | None = None,
    ) -> dict[str, Any]:
        to = self._normalize_address(to, "recipient")
        cc_norm = self._normalize_recipients(cc)
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{self._display_name} <{self._email}>"
        msg["To"] = to
        msg["Subject"] = subject
        if cc_norm:
            msg["Cc"] = cc_norm
        msg.attach(MIMEText(body, "plain"))
        for filename, payload, mime_type in attachments or []:
            maintype, _, subtype = mime_type.partition("/")
            part = MIMEBase(maintype, subtype, method="REQUEST" if subtype == "calendar" else None)
            part.set_payload(payload)
            part.add_header("Content-Disposition", "attachment", filename=filename)
            part.set_charset("utf-8")
            msg.attach(part)

        self._smtp_send(msg)

        logger.info("email_sent", to=to)
        return {"to": to, "subject": subject, "status": "sent"}

    def send_html_message(
        self, to: str, subject: str, html_body: str, cc: str | None = None
    ) -> dict[str, Any]:
        to = self._normalize_address(to, "recipient")
        cc_norm = self._normalize_recipients(cc)
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{self._display_name} <{self._email}>"
        msg["To"] = to
        msg["Subject"] = subject
        if cc_norm:
            msg["Cc"] = cc_norm
        msg.attach(MIMEText(html_body, "html"))

        self._smtp_send(msg)

        logger.info("email_html_sent", to=to)
        return {"to": to, "subject": subject, "status": "sent"}

    # ── IMAP: Read ──

    def _imap_connect(self) -> imaplib.IMAP4_SSL:
        conn = imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=30)
        conn.login(self._email, self._password)
        return conn

    def list_messages(
        self, max_results: int = 10, query: str = ""
    ) -> list[dict[str, Any]]:
        conn = self._imap_connect()
        try:
            conn.select("INBOX", readonly=True)
            search_criteria = query if query else "ALL"
            status, msg_ids = conn.search(None, search_criteria)
            if status != "OK":
                return []

            ids = msg_ids[0].split()
            ids.reverse()
            result: list[dict[str, Any]] = []

            for msg_id in ids[:max_results]:
                try:
                    status, data = conn.fetch(msg_id, "(FLAGS BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
                    if status != "OK":
                        continue
                    raw = data[0][1]
                    msg = email.message_from_bytes(raw) if isinstance(raw, bytes) else email.message_from_string(raw)
                    label_ids = []
                    flags_data = conn.fetch(msg_id, "(FLAGS)")[1][0]
                    if isinstance(flags_data, bytes):
                        flag_str = flags_data.decode(errors="replace")
                        if "\\Seen" not in flag_str:
                            label_ids.append("UNREAD")
                        if "\\Flagged" in flag_str:
                            label_ids.append("STARRED")

                    result.append({
                        "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                        "from": self._decode_header(msg.get("From", "")),
                        "subject": self._decode_header(msg.get("Subject", "(no subject)")),
                        "date": msg.get("Date", ""),
                        "snippet": "",
                        "label_ids": label_ids,
                    })
                except Exception as exc:
                    logger.warning("imap_list_detail_failed", error=str(exc))
            return result
        finally:
            conn.logout()

    def get_message(self, message_id: str) -> dict[str, Any]:
        conn = self._imap_connect()
        try:
            conn.select("INBOX", readonly=True)
            status, data = conn.fetch(
                message_id.encode() if isinstance(message_id, str) else message_id,
                "(RFC822)",
            )
            if status != "OK":
                raise ValueError("Failed to fetch message")

            raw = data[0][1]
            msg = email.message_from_bytes(raw) if isinstance(raw, bytes) else email.message_from_string(raw)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() in ("text/plain", "text/html"):
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            body = payload.decode(charset, errors="replace")
                            if part.get_content_type() == "text/plain":
                                break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")

            return {
                "id": message_id,
                "from": self._decode_header(msg.get("From", "")),
                "to": self._decode_header(msg.get("To", "")),
                "subject": self._decode_header(msg.get("Subject", "")),
                "date": msg.get("Date", ""),
                "body": body,
                "snippet": body[:200] if body else "",
                "label_ids": [],
            }
        finally:
            conn.logout()

    def get_profile(self) -> dict[str, Any]:
        conn = self._imap_connect()
        try:
            conn.select("INBOX", readonly=True)
            status, data = conn.status("INBOX", "(MESSAGES)")
            messages_total = 0
            if status == "OK" and data:
                parts = data[0].decode() if isinstance(data[0], bytes) else str(data[0])
                for part in parts.split():
                    if part.startswith("MESSAGES"):
                        messages_total = int(part.split("MESSAGES")[1])
                        break
            return {"email": self._email, "messages_total": messages_total}
        finally:
            conn.logout()

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        self.close()

    @staticmethod
    def _decode_header(value: str | None) -> str:
        if not value:
            return ""
        parts: list[str] = []
        for text, charset in decode_header(value):
            if isinstance(text, bytes):
                parts.append(text.decode(charset or "utf-8", errors="replace"))
            else:
                parts.append(text)
        return " ".join(parts)


class GmailClient(EmailClient):
    """Backward-compatible alias for EmailClient."""
    pass


def get_gmail_client() -> GmailClient:
    return GmailClient()
