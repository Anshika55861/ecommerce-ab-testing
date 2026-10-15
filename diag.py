import base64
import requests
from src.config import EBAY_APP_ID, EBAY_CERT_ID, OAUTH_URL, OAUTH_SCOPE

print("APP_ID  :", repr(EBAY_APP_ID))
print("APP_ID len :", len(EBAY_APP_ID))
print("CERT_ID :", repr(EBAY_CERT_ID))
print("CERT_ID len:", len(EBAY_CERT_ID))
print()

creds = base64.b64encode(f"{EBAY_APP_ID}:{EBAY_CERT_ID}".encode()).decode()

r = requests.post(
    OAUTH_URL,
    headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {creds}",
    },
    data={"grant_type": "client_credentials", "scope": OAUTH_SCOPE},
)

print("STATUS:", r.status_code)
print("BODY:", r.text)