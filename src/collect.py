"""eBay Browse API collector.

Setup (do this on day 1 — approval is not instant):
  1. Register at https://developer.ebay.com/
  2. Create an application keyset -> you get an App ID (Client ID)
     and a Cert ID (Client Secret)
  3. Copy .env.example to .env and paste them in
  4. python -m src.collect

The Browse API needs an *application* token (client-credentials grant), not a
user token, because we're only reading public listings.

Raw responses are cached to data/raw/ as JSON so you never re-hit the API
while iterating on analysis. Rate limits are real; be kind to them.
"""
import base64
import json
import time
from datetime import datetime

import pandas as pd
import requests

from src import config as C

CACHE_DIR = C.DATA_RAW


def get_access_token() -> str:
    """Client-credentials OAuth. Token is valid ~2 hours."""
    if not C.EBAY_APP_ID or not C.EBAY_CERT_ID:
        raise RuntimeError(
            "Missing credentials. Copy .env.example to .env and fill in "
            "EBAY_APP_ID and EBAY_CERT_ID."
        )
    creds = base64.b64encode(
        f"{C.EBAY_APP_ID}:{C.EBAY_CERT_ID}".encode()
    ).decode()
    resp = requests.post(
        C.OAUTH_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {creds}",
        },
        data={"grant_type": "client_credentials", "scope": C.OAUTH_SCOPE},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search(token: str, query: str, limit: int = 200, offset: int = 0) -> dict:
    """One page of the Browse API item_summary/search endpoint."""
    resp = requests.get(
        C.BROWSE_SEARCH_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": C.MARKETPLACE,
            "Content-Type": "application/json",
        },
        params={"q": query, "limit": limit, "offset": offset},
        timeout=30,
    )
    if resp.status_code == 429:                  # rate limited
        print("      rate limited, sleeping 60s")
        time.sleep(60)
        return search(token, query, limit, offset)
    resp.raise_for_status()
    return resp.json()


def collect_term(token: str, query: str, total: int) -> list[dict]:
    """Paginate through `total` items for one search term, with caching."""
    slug = query.replace(" ", "_")
    cache = CACHE_DIR / f"{slug}.json"
    if cache.exists():
        print(f"      cache hit: {cache.name}")
        return json.loads(cache.read_text())

    items, offset, page_size = [], 0, 200
    while len(items) < total:
        page = search(token, query, limit=page_size, offset=offset)
        batch = page.get("itemSummaries") or []
        if not batch:
            break
        items.extend(batch)
        offset += page_size
        print(f"      {query}: {len(items)} items")
        time.sleep(0.5)                          # be polite
    cache.write_text(json.dumps(items))
    return items


def flatten(items: list[dict], query: str) -> pd.DataFrame:
    """Pull the fields we actually analyse out of the nested JSON."""
    rows = []
    for it in items:
        price = (it.get("price") or {})
        ship = (it.get("shippingOptions") or [{}])[0]
        ship_cost = (ship.get("shippingCost") or {})
        seller = (it.get("seller") or {})
        rows.append({
            "search_term": query,
            "item_id": it.get("itemId"),
            "title": it.get("title"),
            "price": pd.to_numeric(price.get("value"), errors="coerce"),
            "currency": price.get("currency"),
            "condition": it.get("condition"),
            "buying_options": ",".join(it.get("buyingOptions") or []),
            "shipping_cost": pd.to_numeric(ship_cost.get("value"), errors="coerce"),
            "shipping_type": ship.get("shippingCostType"),
            "seller_username": seller.get("username"),
            "seller_feedback_pct": pd.to_numeric(
                seller.get("feedbackPercentage"), errors="coerce"),
            "seller_feedback_score": pd.to_numeric(
                seller.get("feedbackScore"), errors="coerce"),
            "item_location": (it.get("itemLocation") or {}).get("country"),
            "top_rated": it.get("topRatedBuyingExperience"),
            "collected_at": datetime.utcnow().isoformat(),
        })
    return pd.DataFrame(rows)


def main():
    token = get_access_token()
    print("Token acquired.")
    frames = []
    for term in C.SEARCH_TERMS:
        print(f"  Collecting '{term}' ...")
        items = collect_term(token, term, C.ITEMS_PER_TERM)
        frames.append(flatten(items, term))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="item_id")
    df.to_csv(C.LISTINGS_FILE, index=False)
    print(f"\nSaved {len(df):,} listings -> {C.LISTINGS_FILE}")


if __name__ == "__main__":
    main()
