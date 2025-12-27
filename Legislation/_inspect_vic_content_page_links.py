import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

URL = "https://content.legislation.vic.gov.au/site-6/in-force/acts/duties-act-2000"


def main() -> None:
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urljoin(URL, href)
        low = abs_url.lower()
        if "sites/default/files/" not in low:
            continue
        if not any(ext in low for ext in [".pdf", ".docx", ".doc", ".rtf", ".xml"]):
            continue
        label = (a.get_text(" ", strip=True) or "").strip()
        rows.append((label, abs_url))

    print("total file links", len(rows))

    # Show the first 30 in DOM order
    for i, (label, link) in enumerate(rows[:30], start=1):
        print(f"{i:02d}", (label[:80] + "…") if len(label) > 80 else label)
        print("   ", link)

    # Show a few 'authorised' PDFs
    auth = [(label, link) for (label, link) in rows if "authorised" in link.lower() or "authorised" in label.lower()]
    print("\nauthorised candidates", len(auth))
    for i, (label, link) in enumerate(auth[:20], start=1):
        print(f"A{i:02d}", (label[:80] + "…") if len(label) > 80 else label)
        print("   ", link)

    # Show a few .docx candidates
    docx = [(label, link) for (label, link) in rows if link.lower().endswith(".docx")]
    print("\ndocx candidates", len(docx))
    for i, (label, link) in enumerate(docx[:20], start=1):
        print(f"D{i:02d}", (label[:80] + "…") if len(label) > 80 else label)
        print("   ", link)


if __name__ == "__main__":
    main()
