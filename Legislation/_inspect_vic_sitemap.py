import re
import sys
import requests

UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url: str) -> requests.Response:
    return requests.get(url, headers=UA, timeout=30)


def find_in_sitemap_pages(needle: str, max_pages: int = 5) -> list[str]:
    hits: list[str] = []
    for page in range(0, max_pages + 1):
        url = "https://content.legislation.vic.gov.au/sitemap.xml" + (f"?page={page}" if page else "")
        r = fetch(url)
        if r.status_code != 200:
            print("sitemap", url, "->", r.status_code)
            continue
        locs = re.findall(r"<loc>([^<]+)</loc>", r.text)
        for loc in locs:
            if needle.lower() in loc.lower():
                hits.append(loc)
    return hits


def inspect_page(url: str) -> None:
    r = fetch(url)
    print("\nPAGE", url)
    print("status", r.status_code, "ct", r.headers.get("content-type"), "len", len(r.content))
    text = r.text
    for key in [".pdf", ".docx", ".rtf", "download", "attachment", "field", "file", "application/pdf"]:
        print(key, text.lower().count(key.strip('.')))
    # show first few file-like URLs
    urls = sorted(set(re.findall(r"https?://[^\s\"']+", text)))
    fileish = [u for u in urls if any(ext in u.lower() for ext in [".pdf", ".docx", ".rtf", ".xml", ".zip"]) or "sites/default/files" in u.lower()]
    print("fileish_urls", len(fileish))
    for u in fileish[:50]:
        print(u)


def main() -> None:
    needle = sys.argv[1] if len(sys.argv) > 1 else "duties-act-2000"
    hits = find_in_sitemap_pages(needle, max_pages=10)
    print("hits", len(hits))
    for h in hits[:20]:
        print(h)
    for h in hits[:3]:
        inspect_page(h)


if __name__ == "__main__":
    main()
