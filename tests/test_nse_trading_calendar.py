from datetime import date

from nse_trading_calendar import HOLIDAYS, is_nse_trading_day


def test_weekday_not_in_holiday_calendar_is_trading_day():
    assert is_nse_trading_day(date(2026, 11, 6))


def test_weekend_is_not_trading_day():
    assert not is_nse_trading_day(date(2026, 11, 7))


def test_exchange_holiday_is_not_trading_day():
    assert date(2026, 11, 10) in HOLIDAYS
    assert not is_nse_trading_day(date(2026, 11, 10))
