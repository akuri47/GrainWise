"""
Agmarknet Mandi API Integration
================================
Fetches live rice/paddy prices from data.gov.in API.
Falls back to static prices if API unavailable.

Lookup strategy:
  1. Try today's date first
  2. Walk back day by day up to 7 days
  3. If still nothing, return None (caller uses static fallback)
"""

import requests
from datetime import datetime, timedelta

DATA_GOV_IN_API_KEY = "579b464db66ec23bdd00000120d0e2380eaf4e8044f87fd935e185b7"
DATA_GOV_IN_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

VARIETY_MAPPING = {
    "1001":           {"commodity": "Paddy(Common)", "variety": "1001",               "is_other": False},
    "1010":           {"commodity": "Paddy(Common)", "variety": "MTU-1010",           "is_other": False},
    "ir64":           {"commodity": "Paddy(Common)", "variety": "I.R. 64",            "is_other": False},
    "mansuri":        {"commodity": "Paddy(Common)", "variety": "Masuri",             "is_other": False},
    "swarna":         {"commodity": "Paddy(Common)", "variety": "Swarna Masuri (New)","is_other": False},
    "sonam":          {"commodity": "Paddy(Common)", "variety": "Sona Masuri New",    "is_other": False},
    "golden_mansuri": {"commodity": "Paddy(Common)", "variety": "Sona Masuri (OLD)",  "is_other": False},
    "nati_mansuri":   {"commodity": "Paddy(Common)", "variety": "Other",              "is_other": True},
    "ganga_kaveri":   {"commodity": "Paddy(Common)", "variety": "Other",              "is_other": True},
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}


def _fetch_for_variety(variety_name, date_str, limit=20):
    url = (
        f"{DATA_GOV_IN_URL}"
        f"?api-key={DATA_GOV_IN_API_KEY}"
        f"&format=json"
        f"&limit={limit}"
        f"&filters[variety]={variety_name}"
        f"&filters[arrival_date]={date_str}"
        f"&filters[commodity]=Paddy(Common)"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json().get("records", [])
    except Exception:
        pass
    return []


def _parse_records(records):
    seen = set()
    details = []
    for rec in records:
        try:
            price = float(rec.get("modal_price", 0))
            if price <= 1000:
                continue
            state = rec.get("state", "Unknown")
            market = rec.get("market", "Unknown")
            date = rec.get("arrival_date", "")
            key = (state, market, date)
            if key in seen:
                continue
            seen.add(key)
            details.append({"state": state, "market": market, "date": date, "price": price})
        except (ValueError, TypeError):
            continue
    return details


def fetch_live_price(variety_key):
    result = {"price": None, "used_other": False, "api_variety": "", "mandi_details": []}

    mapping = VARIETY_MAPPING.get(variety_key)
    if not mapping:
        return result

    variety_name = mapping["variety"]
    is_other = mapping["is_other"]
    result["used_other"] = is_other
    result["api_variety"] = variety_name

    today = datetime.now()
    for days_back in range(8):
        check_date = today - timedelta(days=days_back)
        date_str = check_date.strftime("%d/%m/%Y")
        records = _fetch_for_variety(variety_name, date_str)
        details = _parse_records(records)

        if details:
            prices = [d["price"] for d in details]
            avg_price = round(sum(prices) / len(prices), 2)
            result["price"] = avg_price
            result["mandi_details"] = details
            label = "today" if days_back == 0 else f"{days_back}d ago"
            print(
                f"  [API] {variety_name}: Rs.{avg_price}/qtl "
                f"({len(details)} mandis, {label}: {date_str})"
            )
            return result

    print(f"  [API] No data for {variety_name} in 7 days. Using static fallback.")
    return result
