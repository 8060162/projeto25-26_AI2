from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from rag_api.reporting.models import Granularity, period_key

LISBON = ZoneInfo("Europe/Lisbon")


def test_realtime_always_same_key():
    t1 = datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 6, 3, 23, 0, tzinfo=timezone.utc)
    assert period_key(Granularity.REALTIME, t1) == period_key(Granularity.REALTIME, t2)


def test_hourly_key_in_lisbon_timezone():
    # 23:30 UTC = 00:30 Lisboa (UTC+1) — chave deve reflectir hora de Lisboa
    t = datetime(2026, 6, 3, 23, 30, tzinfo=timezone.utc)
    assert period_key(Granularity.HOURLY, t) == "2026-06-04T00"


def test_daily_key_in_lisbon_timezone():
    t = datetime(2026, 6, 3, 23, 30, tzinfo=timezone.utc)
    assert period_key(Granularity.DAILY, t) == "2026-06-04"


def test_monthly_key_format():
    t = datetime(2026, 6, 3, 14, 0, tzinfo=timezone.utc)
    assert period_key(Granularity.MONTHLY, t) == "2026-06"