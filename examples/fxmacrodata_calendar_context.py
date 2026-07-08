"""Join Smart Money Concepts output with FXMacroData macro-event context."""

import json
from datetime import date, timedelta
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BASE_URL = "https://fxmacrodata.com/api/v1/calendar/{currency}"


def fetch_release_calendar(currency="USD", start_date=None, end_date=None):
    today = date.today()
    start_date = start_date or today.isoformat()
    end_date = end_date or (today + timedelta(days=30)).isoformat()
    query = urlencode({"start_date": start_date, "end_date": end_date})

    with urlopen(f"{BASE_URL.format(currency=currency)}?{query}", timeout=20) as response:
        payload = json.load(response)

    return pd.DataFrame(payload.get("data", []))


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
