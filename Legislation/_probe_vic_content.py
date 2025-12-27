import requests

URLS = [
    "https://content.legislation.vic.gov.au",
    "https://content.legislation.vic.gov.au/",
    "https://content.legislation.vic.gov.au/graphql",
    "https://content.legislation.vic.gov.au/api",
    "https://content.legislation.vic.gov.au/api/",
    "https://content.legislation.vic.gov.au/openapi.json",
    "https://content.legislation.vic.gov.au/swagger.json",
    "https://content.legislation.vic.gov.au/robots.txt",
    "https://content.legislation.vic.gov.au/sitemap.xml",
]


def main() -> None:
    for url in URLS:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20, allow_redirects=True)
            ct = r.headers.get("content-type")
            print("\n", url, "->", r.status_code, "final", r.url)
            print("ct:", ct)
            print("len:", len(r.content))
            sample = r.text[:300].replace("\n", " ").replace("\r", " ")
            print("sample:", sample)
        except Exception as e:
            print("\n", url, "-> ERROR", repr(e))


if __name__ == "__main__":
    main()
