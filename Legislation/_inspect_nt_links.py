import re
import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "https://legislation.nt.gov.au/Legislation/FIRST-HOME-OWNER-GRANT-ACT-2000"
html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text

api = re.findall(r"/api/sitecore/Act/[^\"']+", html, flags=re.IGNORECASE)
hrefs = re.findall(r'href="([^"]+)"', html)

filtered = [h for h in hrefs if any(x in h.lower() for x in ("doc", "word", "pdf", "rtf"))]

print("URL", url)
print("HTML_LEN", len(html))
print("API_COUNT", len(api))
print("API_SAMPLE")
for a in api[:50]:
    print(" ", a)
print("HREF_COUNT", len(hrefs))
print("FILTERED_HREF_COUNT", len(filtered))
print("FILTERED_HREF_SAMPLE")
for h in filtered[:80]:
    print(" ", h)
