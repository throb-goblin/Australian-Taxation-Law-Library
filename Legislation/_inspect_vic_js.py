import re
import requests

base = 'https://www.legislation.vic.gov.au'
js_path = '/_nuxt/DFMlZTju.js'
resp = requests.get(base + js_path, headers={'User-Agent':'Mozilla/5.0'}, timeout=30)
print('status', resp.status_code, 'len', len(resp.content))
text = resp.text

# Look for obvious API base URLs
urls = set(re.findall(r"https?://[^\s\"']+", text))
apiish = [u for u in urls if any(k in u.lower() for k in ['api', 'graphql', 'content.legislation', 'elastic'])]
print('apiish_urls', len(apiish))
for u in sorted(apiish)[:80]:
    print(u)

# Look for embedded paths that look like API routes
paths = set(re.findall(r"/(?:api|graphql)[^\"']{0,120}", text))
print('apiish_paths', len(paths))
for p in sorted(paths)[:120]:
    print(p)

# Look for file extensions references
for ext in ['.pdf', '.docx', '.rtf']:
    print(ext, text.lower().count(ext.strip('.')))

# Try to locate strings related to downloads
for kw in ['download', 'document', 'attachment', 'pdf']:
    print(kw, text.lower().count(kw))
