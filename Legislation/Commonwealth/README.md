# Commonwealth sync bot

This folder contains a self-contained Python sync bot that maintains a plain-text offline library.

Notes:

- If `title_id` is present, the bot attempts a best-effort API version check when available.
- If the API is unreachable, it falls back to URL/HTML detection.

## Run

From the repo root:

```powershell
cd "Commonwealth"
python -m bot.sync
```

## Common options

```powershell
python -m bot.sync --user-agent "YourBotName/1.0 (contact@example.com)" --sleep-seconds 1.0 --max-retries 4
```

Output: `data/<library_id>.txt` (overwritten on update)
