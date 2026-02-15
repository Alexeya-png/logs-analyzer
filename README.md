# WotR Tool (Logs & Reports)

GUI utility for War of the Ring (Tabletop Simulator) logs.

Features:
- Analyze a single log file or a local folder of logs (auto-picks the last/full log per game and removes duplicates).
- Download game reports from https://waroftheringcommunity.net/game-reports for a date range (optional user filter).
- Compute overall “luck” statistics from downloaded site logs:
  - Luckiest SP (most 6s)
  - Luckiest FP (most 6s)
  - Top 10 luckiest overall (luck value only)
  - Top 10 unluckiest overall (luck value only)

Luck definition:
- `luck = six_count - (total_dice / 6)`
- Positive = more 6s than expected, negative = fewer.

## Requirements

- Python 3.10+ (3.11 recommended)
- Windows / Linux / macOS
- Playwright is required only for “Download reports”

## Install

```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -U pip
pip install playwright
playwright install chromium
```

If you only want to analyze already existing logs (no downloading), you can skip Playwright installation.

## Run

```bash
python wotr_tool.py
```

## UI Overview

Single window with 3 sections.

### 1) Log statistics (file or folder)
- Choose a log file OR a folder with logs.
- Optional: “Only player” field to print stats for one player only.
- For local folders, the tool:
  - sorts logs by timestamp (from filename when possible, otherwise filesystem time),
  - groups logs that belong to the same match,
  - keeps only the most complete log per group,
  - removes duplicate matches by content signature.

Output format per player:
```text
PlayerName rolled 190 combat dice, expected 31,67 per face
 1: 32 (0)
 2: 43 (+11)
 ...
 6 total: 24
```

### 2) Download reports from website
Downloads reports for a date range (optionally filtered by a username appearing in the row text).

Files are saved to the selected output folder using this naming pattern:
```text
YYYY-MM-DD_HHMMSS_FP_<FP name>_SP_<SP name>.log
```

The tool maintains `downloaded_ids.json` in the output folder to avoid re-downloading already seen report IDs.

### 3) Overall statistics (downloaded site logs)
Pick the folder where site logs were downloaded (default: `site_logs/`).

Output includes:
- Luckiest SP (most 6)
- Luckiest FP (most 6)
- Top 10 luckiest (by luck value)
- Top 10 unluckiest (by luck value)

Top lists format:
```text
Top 10 unluckiest
1. Alexey'a – -34,67
2. Jaymanen – -30,67
...
```

## Suggested repo layout

```text
repo/
  wotr_tool.py
  site_logs/               # downloaded logs
  downloaded_ids.json      # created inside the site_logs folder
```

## Troubleshooting

### Playwright error: “Executable doesn't exist…”
Install browser binaries:
```bash
playwright install chromium
```

### Empty stats / no rolls detected
The parser counts dice rolls from lines like:
```text
<game> PlayerName<~Controls.1~> 1 5 3 6 4 null
```

“Interpreter” and “Talker” pseudo-players are ignored.

## License
MIT (or choose your own)
