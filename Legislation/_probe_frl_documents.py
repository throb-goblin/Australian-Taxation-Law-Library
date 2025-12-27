import sys
import urllib.parse

import requests


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else "C2004A00454"
    base = "https://api.prod.legislation.gov.au/v1/Documents"

    filters = [
        f"titleId eq '{title}' and type eq 'Primary' and format eq 'Pdf' and isAuthorised eq true",
        f"titleId eq '{title}' and type eq 'Primary' and format eq 'Pdf'",
        f"titleId eq '{title}' and format eq 'Pdf'",
        f"titleId eq '{title}' and type eq 'Primary' and format eq 'Word'",
    ]

    for flt in filters:
        params = {"$top": "1", "$orderby": "start desc", "$filter": flt}
        url = base + "?" + urllib.parse.urlencode(params, safe="()'=:$, ")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        print("\nfilter:", flt)
        print("status", r.status_code, "ct", r.headers.get("content-type"), "len", len(r.content))
        try:
            j = r.json()
            v = j.get("value") if isinstance(j, dict) else None
            print("items", 0 if not v else len(v))
            if v:
                it = v[0]
                print(
                    "format", it.get("format"),
                    "type", it.get("type"),
                    "isAuthorised", it.get("isAuthorised"),
                    "start", it.get("start"),
                    "registerId", it.get("registerId"),
                    "compilationNumber", it.get("compilationNumber"),
                )
        except Exception as e:
            print("json_err", repr(e))


if __name__ == "__main__":
    main()
