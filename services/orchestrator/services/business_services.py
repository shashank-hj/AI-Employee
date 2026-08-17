import uuid
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


class OrderService:
    async def lookup_order(self, order_id: str) -> dict:
        order_id_clean = order_id.strip().upper()
        return {
            "order_id": order_id_clean,
            "status": "processing",
            "total_amount": 4999.0,
            "currency": "INR",
            "items": [{"name": "Product", "quantity": 1, "price": 4999.0}],
            "tracking_number": f"TRK-{uuid.uuid4().hex[:8].upper()}",
            "courier": "BlueDart",
            "estimated_delivery": datetime.now(timezone.utc).isoformat(),
        }

    async def health_check(self) -> bool:
        return True


class CalendarService:
    async def get_availability(self, query: str | None = None, days_ahead: int = 5) -> list[dict]:
        import random
        from datetime import timedelta
        slots = []
        base_date = datetime.now(timezone.utc).date()
        for day_offset in range(days_ahead):
            day = base_date + timedelta(days=day_offset)
            date_str = day.strftime("%Y-%m-%d")
            for hour in range(9, 17):
                if random.random() > 0.3:
                    slots.append({
                        "time": f"{hour:02d}:00",
                        "date": date_str,
                        "available": True,
                        "duration_minutes": 60,
                    })
        return slots

    async def schedule_meeting(self, title: str, date: str, time: str, attendees: list[str] | None = None) -> dict:
        meeting_id = f"mtg-{uuid.uuid4().hex[:8]}"
        return {
            "meeting_id": meeting_id,
            "title": title,
            "date": date,
            "time": time,
            "attendees": attendees or [],
            "status": "scheduled",
            "link": f"https://meet.ai-employee.com/{meeting_id}",
        }

    async def schedule_demo(self, title: str, attendees: list[str], date: str, time: str, duration: int = 30) -> dict:
        meeting_id = f"demo-{uuid.uuid4().hex[:8]}"
        return {
            "meeting_id": meeting_id,
            "title": title,
            "date": date,
            "time": time,
            "attendees": attendees,
            "duration_minutes": duration,
            "status": "scheduled",
            "link": f"https://meet.ai-employee.com/{meeting_id}",
        }

    async def health_check(self) -> bool:
        return True


class PricingService:
    _PRICING_TIERS = [
        {
            "tier": "Free",
            "price_monthly": 0,
            "price_yearly": 0,
            "features": ["Basic support", "5 queries/day", "Email notifications"],
        },
        {
            "tier": "Pro",
            "price_monthly": 2999,
            "price_yearly": 29990,
            "features": ["Priority support", "Unlimited queries", "API access", "Custom integrations"],
        },
        {
            "tier": "Enterprise",
            "price_monthly": 9999,
            "price_yearly": 99990,
            "features": ["Dedicated support", "Unlimited everything", "API access", "Custom integrations", "SLA guarantee", "On-premise option"],
        },
        {
            "tier": "Pro Annual (INR)",
            "price_monthly": 2499,
            "price_yearly": 24990,
            "features": ["Priority support", "Unlimited queries", "API access", "Annual billing discount"],
        },
    ]

    async def search_pricing(self, query: str, top_k: int = 5) -> list[dict]:
        q = query.lower()
        results = []
        for tier in self._PRICING_TIERS:
            if q in tier["tier"].lower() or any(q in f.lower() for f in tier["features"]):
                results.append(tier)
        if not results:
            results = self._PRICING_TIERS[:3]
        return results[:top_k]

    async def get_tier(self, tier_name: str) -> dict | None:
        for t in self._PRICING_TIERS:
            if t["tier"].lower() == tier_name.lower():
                return t
        return None

    async def health_check(self) -> bool:
        return True


class EmailService:
    async def send_email(self, to: str, subject: str, body: str) -> dict:
        message_id = f"MSG-{uuid.uuid4().hex[:8].upper()}"
        logger.info("email_sent", message_id=message_id, to=to, subject=subject)
        return {
            "message_id": message_id,
            "to": to,
            "subject": subject,
            "status": "sent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def health_check(self) -> bool:
        return True


class WeatherService:
    _CITY_DATA = {
        "bengaluru": {"temp_range": (22, 32), "conditions": ["Sunny", "Partly Cloudy", "Cloudy", "Rainy"]},
        "bangalore": {"temp_range": (22, 32), "conditions": ["Sunny", "Partly Cloudy", "Cloudy", "Rainy"]},
        "mumbai": {"temp_range": (25, 35), "conditions": ["Humid", "Partly Cloudy", "Rainy", "Sunny"]},
        "delhi": {"temp_range": (28, 42), "conditions": ["Sunny", "Hazy", "Partly Cloudy", "Clear"]},
        "new delhi": {"temp_range": (28, 42), "conditions": ["Sunny", "Hazy", "Partly Cloudy", "Clear"]},
        "chennai": {"temp_range": (28, 38), "conditions": ["Humid", "Sunny", "Partly Cloudy", "Rainy"]},
        "kolkata": {"temp_range": (26, 36), "conditions": ["Humid", "Partly Cloudy", "Rainy", "Sunny"]},
        "hyderabad": {"temp_range": (24, 38), "conditions": ["Sunny", "Partly Cloudy", "Clear", "Rainy"]},
        "pune": {"temp_range": (20, 34), "conditions": ["Sunny", "Partly Cloudy", "Clear", "Rainy"]},
        "jaipur": {"temp_range": (25, 42), "conditions": ["Sunny", "Dry", "Clear", "Hazy"]},
    }
    _DEFAULT_RANGE = (20, 35)

    async def get_weather(self, city: str, unit: str = "celsius") -> dict:
        import random
        city_key = city.lower().strip()
        city_data = self._CITY_DATA.get(city_key, {"temp_range": self._DEFAULT_RANGE, "conditions": ["Sunny", "Partly Cloudy", "Cloudy", "Clear"]})
        lo, hi = city_data["temp_range"]
        temperature = random.randint(lo, hi)
        return {
            "city": city,
            "location": city,
            "temperature": temperature,
            "unit": unit,
            "conditions": random.choice(city_data["conditions"]),
            "humidity": random.randint(40, 85),
            "wind_speed": round(random.uniform(5, 25), 1),
        }

    async def health_check(self) -> bool:
        return True


class EscalationService:
    async def transfer_to_human(self, user_input: str, reason: str, priority: str = "NORMAL") -> dict:
        ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
        logger.info("escalation_created", ticket_id=ticket_id, reason=reason, priority=priority)
        return {
            "ticket_id": ticket_id,
            "status": "open",
            "priority": priority,
            "reason": reason,
            "assigned_team": "Support",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    async def health_check(self) -> bool:
        return True
