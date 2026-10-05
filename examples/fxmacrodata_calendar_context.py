"""Join Smart Money Concepts output with FXMacroData macro-event context."""

import json
from datetime import date, timedelta
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BASE_URL = "https://api.fxmacrodata.com/v1/calendar/{currency}"


def fetch_release_calendar(currency="USD", start_date=None, end_date=None):
    today = date.today()
    start_date = start_date or today.isoformat()
    end_date = end_date or (today + timedelta(days=30)).isoformat()
    query = urlencode({"start_date": start_date, "end_date": end_date})

    with urlopen(f"{BASE_URL.format(currency=currency)}?{query}", timeout=20) as response:
        payload = json.load(response)

    events = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        detail = payload.get("detail") if isinstance(payload, dict) else None
        raise ValueError(f"FXMacroData returned an unexpected response: {detail or 'missing data list'}")
    calendar = pd.DataFrame([event for event in events if isinstance(event, dict)])
    for column in ("top_tier_for_currency", "market_tier", "announcement_datetime_utc", "date"):
        if column not in calendar.columns:
            calendar[column] = None
    return calendar


def top_tier_release_dates(calendar):
    if calendar.empty:
        return set()

    top_tier = calendar[
        (calendar["top_tier_for_currency"] == True)  # noqa: E712
        | (calendar["market_tier"] == 1)
    ].copy()
    top_tier["release_date"] = top_tier["announcement_datetime_utc"].fillna(top_tier["date"]).str[:10]
    return set(top_tier["release_date"].dropna())


if __name__ == "__main__":
    releases = fetch_release_calendar(start_date="2026-07-01", end_date="2026-07-20")
    print(sorted(top_tier_release_dates(releases)))
