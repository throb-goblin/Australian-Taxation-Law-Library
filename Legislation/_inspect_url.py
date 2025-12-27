import re
import requests

url = 'https://www.legislation.vic.gov.au/in-force/acts/duties-act-2000/'
resp = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=30)
print('status', resp.status_code)
print('final', resp.url)
html = resp.text
print('len', len(html))
for pat in ['.pdf', '.docx', '.rtf', 'download', '__NEXT_DATA__', '_next']:
    print(pat, html.lower().count(pat.strip('.')))
mp = re.search(r"[^\s\"']+\.pdf[^\s\"']*", html, re.I)
print('first_pdf', mp.group(0) if mp else None)
print('has_next_data', '__NEXT_DATA__' in html)
# Print a few lines containing 'href' and 'download'
lines = [ln for ln in html.splitlines() if 'download' in ln.lower() or '.pdf' in ln.lower()][:10]
print('sample_lines', len(lines))
for ln in lines[:10]:
    print(ln[:200])

print("\nembedded_urls")
urls = set(re.findall(r"https?://[^\s\"']+", html))
for u in sorted(urls)[:50]:
    print(u)

print("\nembedded_paths")
paths = set(re.findall(r"(?:src|href)=\"([^\"]+)\"", html))
for p in sorted(paths)[:80]:
    if any(x in p.lower() for x in ["nuxt", "api", "json", "download", "pdf", "xml"]):
        print(p)

print("\ncontains_strings")
for s in ["/api/", "graphql", "content.legislation", "download", "pdf", "document"]:
    print(s, (s.lower() in html.lower()))
