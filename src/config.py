import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
for _d in (DATA_RAW, DATA_PROCESSED, FIGURES, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# --- eBay API credentials. NEVER hardcode these. Put them in .env ---
EBAY_APP_ID = os.getenv("EBAY_APP_ID")
EBAY_CERT_ID = os.getenv("EBAY_CERT_ID")
EBAY_ENV = os.getenv("EBAY_ENV", "production")   # or "sandbox"

_BASE = {
    "production": "https://api.ebay.com",
    "sandbox": "https://api.sandbox.ebay.com",
}
API_BASE = _BASE[EBAY_ENV]
OAUTH_URL = f"{API_BASE}/identity/v1/oauth2/token"
BROWSE_SEARCH_URL = f"{API_BASE}/buy/browse/v1/item_summary/search"
OAUTH_SCOPE = "https://api.ebay.com/oauth/api_scope"
MARKETPLACE = os.getenv("EBAY_MARKETPLACE", "EBAY_US")

SEARCH_TERMS = [
    "wireless earbuds",
    "mechanical keyboard",
    "running shoes",
    "smart watch",
    "dslr camera",
]
ITEMS_PER_TERM = 400          # paginated at 200/page
LISTINGS_FILE = DATA_PROCESSED / "listings.csv"

RANDOM_SEED = 42
ALPHA = 0.05                  # significance level, fixed BEFORE looking at data
POWER = 0.80
