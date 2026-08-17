"""Async persistence for meetings and pending booking proposals (Postgres)."""

from datetime import datetime, timedelta, timezone as tz
from typing import Any

from sqlalchemy import delete, select

from orchestrator.database.session import async_session
from orchestrator.models.calendar_meeting import CalendarMeeting, PendingCalendarBooking


class CalendarMeetingRepository:
    def __init__(self, session_factory=async_session) -> None:
        self._session_factory = session_factory

    async def create(self, meeting: CalendarMeeting) -> CalendarMeeting:
        async with self._session_factory() as db:
            db.add(meeting)
            await db.commit()
            await db.refresh(meeting)
            return meeting

    async def get(self, meeting_id: str) -> CalendarMeeting | None:
        async with self._session_factory() as db:
            return await db.get(CalendarMeeting, meeting_id)

    async def list_meetings(
        self,
        *,
        session_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[CalendarMeeting]:
        stmt = select(CalendarMeeting).order_by(CalendarMeeting.start_at)
        if session_id:
            stmt = stmt.where(CalendarMeeting.session_id == session_id)
        if start:
            stmt = stmt.where(CalendarMeeting.start_at >= start)
        if end:
            stmt = stmt.where(CalendarMeeting.start_at <= end)
        if status:
            stmt = stmt.where(CalendarMeeting.status == status)
        stmt = stmt.limit(limit)
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            return list(result.scalars().all())

    async def find_duplicate(
        self,
        start_at: datetime,
        attendees: list[str],
        window_minutes: int = 15,
    ) -> CalendarMeeting | None:
        lo = start_at - timedelta(minutes=window_minutes)
        hi = start_at + timedelta(minutes=window_minutes)
        stmt = (
            select(CalendarMeeting)
            .where(CalendarMeeting.start_at >= lo)
            .where(CalendarMeeting.start_at <= hi)
            .where(CalendarMeeting.status == "scheduled")
        )
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            for meeting in result.scalars().all():
                if set(meeting.attendees or []) == set(attendees):
                    return meeting
        return None

    async def get_by_provider_event_id(self, provider_event_id: str) -> CalendarMeeting | None:
        stmt = select(CalendarMeeting).where(
            CalendarMeeting.provider_event_id == provider_event_id
        )
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            return result.scalar_one_or_none()

    async def update(self, meeting_id: str, **fields: Any) -> CalendarMeeting | None:
        async with self._session_factory() as db:
            meeting = await db.get(CalendarMeeting, meeting_id)
            if meeting is None:
                return None
            for key, value in fields.items():
                setattr(meeting, key, value)
            meeting.updated_at = datetime.now(tz.utc)
            await db.commit()
            await db.refresh(meeting)
            return meeting

    async def cancel(self, meeting_id: str) -> CalendarMeeting | None:
        return await self.update(
            meeting_id,
            status="cancelled",
            cancelled_at=datetime.now(tz.utc),
        )

    async def delete(self, meeting_id: str) -> bool:
        async with self._session_factory() as db:
            result = await db.execute(
                delete(CalendarMeeting).where(CalendarMeeting.id == meeting_id)
            )
            await db.commit()
            return result.rowcount > 0


class PendingBookingRepository:
    def __init__(self, session_factory=async_session) -> None:
        self._session_factory = session_factory

    async def upsert(
        self,
        session_id: str,
        user_id: str | None,
        title: str,
        start_at: datetime,
        end_at: datetime,
        timezone: str,
        attendees: list[str],
        description: str | None,
        expires_in_minutes: int = 30,
    ) -> PendingCalendarBooking:
        expires_at = datetime.now(tz.utc) + timedelta(minutes=expires_in_minutes)
        async with self._session_factory() as db:
            existing = await db.execute(
                select(PendingCalendarBooking).where(
                    PendingCalendarBooking.session_id == session_id
                )
            )
            proposal = existing.scalar_one_or_none()
            if proposal is None:
                proposal = PendingCalendarBooking(session_id=session_id)
                db.add(proposal)
            proposal.user_id = user_id
            proposal.title = title
            proposal.start_at = start_at
            proposal.end_at = end_at
            proposal.timezone = timezone
            proposal.attendees = attendees
            proposal.description = description
            proposal.status = "proposed"
            proposal.expires_at = expires_at
            proposal.created_at = datetime.now(tz.utc)
            await db.commit()
            await db.refresh(proposal)
            return proposal

    async def get_active(self, session_id: str) -> PendingCalendarBooking | None:
        now = datetime.now(tz.utc)
        stmt = (
            select(PendingCalendarBooking)
            .where(PendingCalendarBooking.session_id == session_id)
            .where(PendingCalendarBooking.status == "proposed")
        )
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            proposal = result.scalar_one_or_none()
            if proposal is None:
                return None
            if proposal.expires_at is not None and proposal.expires_at < now:
                await db.delete(proposal)
                await db.commit()
                return None
            return proposal

    async def clear(self, session_id: str) -> None:
        async with self._session_factory() as db:
            result = await db.execute(
                delete(PendingCalendarBooking).where(
                    PendingCalendarBooking.session_id == session_id
                )
            )
            await db.commit()
            return result.rowcount or 0
