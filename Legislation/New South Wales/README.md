# New South Wales sync bot

This folder contains a self-contained Python sync bot that maintains a plain-text offline library.

NSW specifics:

- Computes `version_id` as a `YYYY-MM-DD` date based on the “Current version for … to date” indicator (and other accepted patterns).
- Prefers NSW XML export endpoints where available; falls back to HTML/PDF.

## Run

From the repo root:

```powershell
cd "New South Wales"
python -m bot.sync
```

## Common options

```powershell
python -m bot.sync --user-agent "YourBotName/1.0 (contact@example.com)" --sleep-seconds 1.0 --max-retries 4
```

Output: `data/<library_id>.txt` (overwritten on update)
