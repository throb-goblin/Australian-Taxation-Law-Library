import re
import sys

import requests


def main() -> None:
    u = sys.argv[1] if len(sys.argv) > 1 else "https://www.legislation.tas.gov.au/view/html/inforce/current/sr-2021-037"
    r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    print("status", r.status_code, "ct", r.headers.get("content-type"), "len", len(r.content))

    html = (r.text or "").lower()
    for pat in [".pdf", "pdf", "/view/pdf", "/download", "rtf", "xml", "application/pdf"]:
        print(pat, html.count(pat.strip(".")))

    hrefs = re.findall(r'href="([^"]+)"', r.text or "", flags=re.I)
    fileish = [h for h in hrefs if any(x in h.lower() for x in ["pdf", "xml", "rtf", "download"]) ]
    print("fileish_hrefs", len(fileish))
    for h in fileish[:80]:
        print(h[:240])


if __name__ == "__main__":
    main()
