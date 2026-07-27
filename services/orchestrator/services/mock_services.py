import random
from datetime import datetime, timedelta
from typing import Any


class MockOrderService:
    async def lookup_order(self, order_id: str) -> dict[str, Any]:
        oid = order_id.strip().upper()
        if not oid:
            return {"success": False, "error": "Order ID is required"}

        statuses = ["processing", "shipped", "in_transit", "delivered", "delayed"]
        # deterministic based on order ID hash so same ID returns same status
        seed = sum(ord(c) for c in oid)
        random.seed(seed)
        status = statuses[seed % len(statuses)]
        delivery_date = (datetime.now() + timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d")

        return {
            "success": True,
            "data": {
                "order_id": oid,
                "status": status,
                "item_count": random.randint(1, 5),
                "total_amount": round(random.uniform(499, 49999), 2),
                "currency": "INR",
                "estimated_delivery": delivery_date,
                "tracking_number": f"TRK{random.randint(100000, 999999)}",
                "courier": random.choice(["BlueDart", "Delhivery", "FedEx", "DTDC"]),
            },
        }


class MockCalendarService:
    async def get_availability(self, query: str, days_ahead: int = 5) -> list[dict[str, Any]]:
        today = datetime.now()
        slots: list[dict[str, Any]] = []
        random.seed(sum(ord(c) for c in query))

        for day_offset in range(1, days_ahead + 1):
            date = today + timedelta(days=day_offset)
            if date.weekday() >= 5:
                continue  # skip weekends
            day_slots = []
            for hour in [9, 10, 11, 14, 15, 16]:
                if random.random() > 0.4:
                    day_slots.append(f"{hour:02d}:00")
                    day_slots.append(f"{hour:02d}:30")
            if day_slots:
                slots.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "day_name": date.strftime("%A"),
                    "available_slots": sorted(day_slots),
                })
        return slots

    async def schedule_demo(
        self,
        title: str,
        attendees: list[str],
        date: str,
        time: str,
        duration_minutes: int = 30,
    ) -> dict[str, Any]:
        meeting_id = f"DEMO-{random.randint(1000, 9999)}"
        return {
            "success": True,
            "data": {
                "meeting_id": meeting_id,
                "title": title,
                "attendees": attendees,
                "datetime": f"{date}T{time}:00",
                "duration_minutes": duration_minutes,
                "status": "scheduled",
                "calendar_link": f"https://calendar.example.com/meeting/{meeting_id}",
            },
        }


class MockPricingService:
    _PLANS = [
        {
            "tier": "Free",
            "monthly_price": 0,
            "annual_price": 0,
            "features": ["Up to 5 agents", "Basic analytics", "Email support", "Community forum"],
            "best_for": "Individuals and small projects",
        },
        {
            "tier": "Pro",
            "monthly_price": 2999,
            "annual_price": 29990,
            "features": ["Up to 20 agents", "Advanced analytics", "Priority email support", "Custom workflows", "API access"],
            "best_for": "Growing teams",
        },
        {
            "tier": "Enterprise",
            "monthly_price": 0,
            "annual_price": 0,
            "features": ["Unlimited agents", "Dedicated support", "SLA guarantees", "Custom integrations", "SSO & SAML", "Audit logs", "On-premise deployment"],
            "best_for": "Large organizations — contact sales for pricing",
        },
    ]

    async def search_pricing(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q = query.lower() if query else ""
        scored: list[dict[str, Any]] = []
        for plan in self._PLANS:
            content = f"{plan['tier']} {', '.join(plan['features'])} {plan['best_for']}".lower()
            score = 0.95 if q and q in content else 0.3
            # boost exact tier name matches
            if plan["tier"].lower() in q:
                score = 1.0
            scored.append({**plan, "score": round(score, 3)})
        scored.sort(key=lambda p: p["score"], reverse=True)
        return scored[:top_k]
