# wotr_tool.py
# Requirements:
# - UI in English
# - Single page (no tabs)
# - No "count whole file" checkbox
# - No headful/maxsteps/sleep controls
# - Downloaded reports keep names: YYYY-MM-DD_HHMMSS_FP_<name>_SP_<name>.log
# - Overall stats: Top 10 unluckiest/luckiest show only luck value like: "1. Name – -34,67"
# - Overall stats: ignore players with < 20 matches

import os
import re
import json
import time
import hashlib
import threading
import queue
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# -----------------------------
# Log parsing
# -----------------------------

TALKER_MARK = "<~Talker.31~>"
ROLL_RE = re.compile(r"^<game>\s+(.+?)<~Controls\.1~>\s*(.*?)\s*null\b")
DICE_RE = re.compile(r"\b[1-6]\b")

IGNORED_PLAYER_RE = re.compile(r"^~(Interpreter|Talker)\.\d+~?$", re.IGNORECASE)

# local logs: wotr2026_1_30_18_14-1-1.log
FNAME_TS_RE = re.compile(r"^wotr(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})", re.IGNORECASE)
TRAIL_SUFFIX_RE = re.compile(r"(-\d+)+$")

# site logs: 2026-02-15_061430_FP_Stella Rossa_SP_Feanor (2).log
SITE_DUP_RE = re.compile(r"\s*\(\d+\)\s*$", re.UNICODE)
SITE_DT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{6})_(.+)$", re.IGNORECASE)

# extracting time + names from row text / suggested filename
ROW_TIME_HHMMSS_RE = re.compile(r"\b(\d{2}):(\d{2}):(\d{2})\b")
ROW_TIME_HHMM_RE = re.compile(r"\b(\d{2}):(\d{2})\b")
ROW_TIME_6DIG_RE = re.compile(r"\b(\d{6})\b")

DATE_DDMMYYYY_RE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")
DATE_YYYYMMDD_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DATE_DDMMYYYY_DOT_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")


def is_ignored_player(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return True
    if IGNORED_PLAYER_RE.match(n):
        return True
    if n.startswith("~") and n.endswith("~"):
        return True
    return False


def norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canon_key(s: str) -> str:
    s = norm_name(s)
    return re.sub(r"[^a-z0-9]+", "", s)


def read_log_lines(path: Path) -> list[str]:
    data = path.read_bytes()
    candidates = []
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        txt = data.decode(enc, errors="replace")
        candidates.append((txt.count("�"), txt))
    _, txt = min(candidates, key=lambda t: t[0])
    return txt.splitlines()


def find_talker_index(lines: list[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        if TALKER_MARK in line:
            return i
    return None


def count_rolls(lines_after: list[str]) -> tuple[dict[str, Counter], list[str]]:
    counts = defaultdict(lambda: Counter({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}))
    roll_lines = []
    for line in lines_after:
        m = ROLL_RE.search(line)
        if not m:
            continue
        player = m.group(1).strip()
        if is_ignored_player(player):
            continue
        dice_part = m.group(2)
        ds = DICE_RE.findall(dice_part)
        if not ds:
            continue
        roll_lines.append(f"{player}:{' '.join(ds)}")
        for d in ds:
            counts[player][int(d)] += 1
    return counts, roll_lines


def fmt_diff(x: int) -> str:
    return f"+{x}" if x > 0 else f"{x}"


def build_player_block(player: str, c: Counter) -> str:
    total = sum(c.values())
    if total <= 0:
        return ""
    exp = total / 6.0
    exp_str = f"{exp:.2f}".replace(".", ",")
    out = [f"{player} rolled {total} combat dice, expected {exp_str} per face"]
    for face in range(1, 7):
        diff = int(round(c[face] - exp))
        out.append(f" {face}: {c[face]} ({fmt_diff(diff)})")
    out.append(f" 6 total: {c[6]}")
    out.append("")
    return "\n".join(out)


def build_text_report(counts: dict[str, Counter], only_player: str = "") -> str:
    only_player = (only_player or "").strip().lower()

    items = []
    for player, c in counts.items():
        if is_ignored_player(player):
            continue
        total = sum(c.values())
        if total <= 0:
            continue
        if only_player and player.lower() != only_player:
            continue
        items.append((total, player, c))
    items.sort(reverse=True)

    if not items:
        return "No dice rolls found\n"

    out = []
    for _, player, c in items:
        out.append(build_player_block(player, c))
    return "\n".join(out).rstrip() + "\n"


def extract_base_and_dt(p: Path) -> tuple[str, Optional[datetime]]:
    stem = p.stem
    base = TRAIL_SUFFIX_RE.sub("", stem)
    m = FNAME_TS_RE.match(base)
    if not m:
        return base, None
    y, mo, d, h, mi = map(int, m.groups())
    try:
        return base, datetime(y, mo, d, h, mi)
    except Exception:
        return base, None


@dataclass
class LogPick:
    path: Path
    base: str
    dt: Optional[datetime]
    total_dice: int
    size: int
    mtime: float
    sig: str
    counts: dict[str, Counter]


def analyze_single_log(path: Path, full_file: bool) -> LogPick:
    lines = read_log_lines(path)

    if full_file:
        lines_after = lines
    else:
        idx = find_talker_index(lines)
        lines_after = lines[idx + 1:] if idx is not None else lines

    counts, roll_lines = count_rolls(lines_after)
    total_dice = sum(sum(c.values()) for c in counts.values())

    sig_src = "\n".join(roll_lines).encode("utf-8")
    sig = hashlib.md5(sig_src).hexdigest()

    base, dt = extract_base_and_dt(path)
    st = path.stat()
    return LogPick(
        path=path,
        base=base,
        dt=dt,
        total_dice=total_dice,
        size=st.st_size,
        mtime=st.st_mtime,
        sig=sig,
        counts=counts,
    )


def pick_best_logs_from_folder(folder: Path) -> list[LogPick]:
    files = []
    for ext in ("*.log", "*.txt"):
        files.extend(folder.glob(ext))

    def sort_key(p: Path):
        _, dt = extract_base_and_dt(p)
        if dt:
            return dt.timestamp()
        try:
            return os.path.getctime(p)
        except Exception:
            return p.stat().st_mtime

    files.sort(key=sort_key)

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        base, _ = extract_base_and_dt(p)
        groups[base].append(p)

    picked_by_base: list[LogPick] = []
    for _, gfiles in groups.items():
        best: Optional[LogPick] = None
        for p in gfiles:
            try:
                info = analyze_single_log(p, full_file=False)
            except Exception:
                continue

            if best is None:
                best = info
                continue
            if info.total_dice > best.total_dice:
                best = info
            elif info.total_dice == best.total_dice:
                if info.size > best.size:
                    best = info
                elif info.size == best.size and info.mtime > best.mtime:
                    best = info

        if best is not None:
            picked_by_base.append(best)

    best_by_sig: dict[str, LogPick] = {}
    for info in picked_by_base:
        prev = best_by_sig.get(info.sig)
        if prev is None:
            best_by_sig[info.sig] = info
            continue
        if info.total_dice > prev.total_dice:
            best_by_sig[info.sig] = info
        elif info.total_dice == prev.total_dice:
            if info.size > prev.size:
                best_by_sig[info.sig] = info
            elif info.size == prev.size and info.mtime > prev.mtime:
                best_by_sig[info.sig] = info

    picked = list(best_by_sig.values())

    def picked_sort_key(info: LogPick):
        if info.dt:
            return info.dt.timestamp()
        try:
            return os.path.getctime(info.path)
        except Exception:
            return info.mtime

    picked.sort(key=picked_sort_key)
    return picked


def merge_counts(picks: list[LogPick]) -> dict[str, Counter]:
    total = defaultdict(lambda: Counter({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}))
    for info in picks:
        for player, c in info.counts.items():
            if is_ignored_player(player):
                continue
            for face in range(1, 7):
                total[player][face] += c[face]
    return total


# -----------------------------
# Site overall stats
# -----------------------------

OVERALL_MIN_MATCHES = 1


@dataclass
class SiteMeta:
    dt: Optional[datetime]
    fp_name: Optional[str]
    sp_name: Optional[str]


def parse_site_meta(p: Path) -> SiteMeta:
    stem = SITE_DUP_RE.sub("", p.stem)
    m = SITE_DT_RE.match(stem)
    if not m:
        return SiteMeta(dt=None, fp_name=None, sp_name=None)

    d_s, t_s, rest = m.group(1), m.group(2), m.group(3)

    dt = None
    try:
        dt = datetime.strptime(d_s + t_s, "%Y-%m-%d%H%M%S")
    except Exception:
        dt = None

    fp_name = None
    sp_name = None

    if rest.startswith("FP_") or rest.startswith("SP_"):
        f1 = rest[:2].upper()
        rest2 = rest[3:]
        i_fp = rest2.find("_FP_")
        i_sp = rest2.find("_SP_")
        candidates = [i for i in (i_fp, i_sp) if i != -1]
        if candidates:
            idx = min(candidates)
            name1 = rest2[:idx]
            f2 = rest2[idx + 1:idx + 3].upper()
            name2 = rest2[idx + 4:]
            if f1 == "FP":
                fp_name = name1
            else:
                sp_name = name1
            if f2 == "FP":
                fp_name = name2
            else:
                sp_name = name2

    return SiteMeta(dt=dt, fp_name=fp_name, sp_name=sp_name)


def site_folder_sorted_files(folder: Path) -> list[Path]:
    files = []
    for ext in ("*.log", "*.txt"):
        files.extend(folder.glob(ext))

    def key(p: Path):
        meta = parse_site_meta(p)
        if meta.dt:
            return meta.dt.timestamp()
        try:
            return os.path.getctime(p)
        except Exception:
            return p.stat().st_mtime

    files.sort(key=key)
    return files


def better_pick(a: LogPick, b: LogPick) -> LogPick:
    if a.total_dice != b.total_dice:
        return a if a.total_dice > b.total_dice else b
    if a.size != b.size:
        return a if a.size > b.size else b
    return a if a.mtime > b.mtime else b


def fmt_luck_plain(x: float) -> str:
    s = f"{abs(x):.2f}".replace(".", ",")
    return f"-{s}" if x < 0 else s


def overall_stats_from_site_folder(folder: Path, dedupe: bool, progress_cb, stop_event: threading.Event) -> str:
    files = site_folder_sorted_files(folder)
    if not files:
        return "No files found\n"

    processed = 0
    keep: dict[str, tuple[LogPick, SiteMeta]] = {}

    for p in files:
        if stop_event.is_set():
            return "Stopped\n"
        try:
            meta = parse_site_meta(p)
            pick = analyze_single_log(p, full_file=True)
        except Exception:
            continue

        processed += 1

        if not dedupe:
            keep[f"{p.as_posix()}::{pick.sig}::{processed}"] = (pick, meta)
        else:
            prev = keep.get(pick.sig)
            if prev is None:
                keep[pick.sig] = (pick, meta)
            else:
                best = better_pick(prev[0], pick)
                if best.path == pick.path:
                    keep[pick.sig] = (pick, meta)

        if processed % 50 == 0:
            progress_cb(f"Read: {processed}")

    picks = list(keep.values())

    def sort_key(item: tuple[LogPick, SiteMeta]):
        meta = item[1]
        if meta.dt:
            return meta.dt.timestamp()
        try:
            return os.path.getctime(item[0].path)
        except Exception:
            return item[0].mtime

    picks.sort(key=sort_key)

    overall = defaultdict(lambda: Counter({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}))
    per_faction = {
        "FP": defaultdict(lambda: Counter({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0})),
        "SP": defaultdict(lambda: Counter({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0})),
    }
    display: dict[str, str] = {}
    match_counts = Counter()

    def luck_score(c: Counter) -> float:
        total = sum(c.values())
        if total <= 0:
            return 0.0
        return c[6] - (total / 6.0)

    for pick, meta in picks:
        norm_counts: dict[str, Counter] = {}
        for pname, c in pick.counts.items():
            if is_ignored_player(pname):
                continue
            n = norm_name(pname)
            if not n:
                continue
            norm_counts[n] = c
            display.setdefault(n, pname)

        key_map = {canon_key(n): n for n in norm_counts.keys() if canon_key(n)}

        def resolve(target_name: Optional[str]) -> Optional[str]:
            if not target_name:
                return None
            t_norm = norm_name(target_name)
            if t_norm in norm_counts:
                return t_norm
            ck = canon_key(t_norm)
            if ck in key_map:
                return key_map[ck]
            return t_norm if t_norm else None

        fp_n = resolve(meta.fp_name)
        sp_n = resolve(meta.sp_name)

        game_players = set()
        if fp_n:
            game_players.add(fp_n)
            display.setdefault(fp_n, meta.fp_name or display.get(fp_n, fp_n))
        if sp_n:
            game_players.add(sp_n)
            display.setdefault(sp_n, meta.sp_name or display.get(sp_n, sp_n))

        if not game_players:
            for n, c in norm_counts.items():
                if sum(c.values()) > 0:
                    game_players.add(n)

        for n in game_players:
            match_counts[n] += 1

        for n, c in norm_counts.items():
            for face in range(1, 7):
                overall[n][face] += c[face]

        if fp_n and fp_n in norm_counts:
            for face in range(1, 7):
                per_faction["FP"][fp_n][face] += norm_counts[fp_n][face]
        if sp_n and sp_n in norm_counts:
            for face in range(1, 7):
                per_faction["SP"][sp_n][face] += norm_counts[sp_n][face]

    rows = []
    for n, c in overall.items():
        if match_counts.get(n, 0) < OVERALL_MIN_MATCHES:
            continue
        total = sum(c.values())
        if total > 0:
            rows.append((luck_score(c), n))

    if not rows:
        return "No data (min matches filter)\n"

    rows_sorted_lucky = sorted(rows, key=lambda t: t[0], reverse=True)
    rows_sorted_unlucky = sorted(rows, key=lambda t: t[0])

    def best_by_most_six(f: str) -> Optional[tuple[str, int, float]]:
        cand = []
        for n, c in per_faction[f].items():
            if match_counts.get(n, 0) < OVERALL_MIN_MATCHES:
                continue
            total = sum(c.values())
            if total <= 0:
                continue
            cand.append((c[6], luck_score(c), n))
        if not cand:
            return None
        cand.sort(reverse=True)
        six, score, n = cand[0]
        return (n, six, score)

    best_sp = best_by_most_six("SP")
    best_fp = best_by_most_six("FP")

    out = []
    out.append(f"Files scanned: {processed} | Games counted: {len(picks)} | Min matches: {OVERALL_MIN_MATCHES}")
    out.append("")

    if best_sp:
        n, six, score = best_sp
        out.append(f"Luckiest SP (most 6): {display.get(n, n)} – 6={six} – {fmt_luck_plain(score)}")
    else:
        out.append("Luckiest SP (most 6): no data")

    if best_fp:
        n, six, score = best_fp
        out.append(f"Luckiest FP (most 6): {display.get(n, n)} – 6={six} – {fmt_luck_plain(score)}")
    else:
        out.append("Luckiest FP (most 6): no data")

    out.append("")
    out.append("Top 10 luckiest")
    for i, (score, n) in enumerate(rows_sorted_lucky[:10], start=1):
        out.append(f"{i}. {display.get(n, n)} – {fmt_luck_plain(score)}")

    out.append("")
    out.append("Top 10 unluckiest")
    for i, (score, n) in enumerate(rows_sorted_unlucky[:10], start=1):
        out.append(f"{i}. {display.get(n, n)} – {fmt_luck_plain(score)}")

    out.append("")
    return "\n".join(out)


# -----------------------------
# Download reports (Playwright)
# -----------------------------

URL = "https://waroftheringcommunity.net/game-reports"
ID_RE = re.compile(r"\b(\d{3,})\b")


@dataclass
class RowInfo:
    report_id: Optional[str]
    played_on: Optional[date]
    raw_text: str


def extract_date_any(txt: str) -> Optional[date]:
    t = " ".join((txt or "").split())

    m = DATE_DDMMYYYY_RE.search(t)
    if m:
        dd, mm, yy = map(int, m.groups())
        try:
            return date(yy, mm, dd)
        except Exception:
            pass

    m = DATE_YYYYMMDD_RE.search(t)
    if m:
        yy, mm, dd = map(int, m.groups())
        try:
            return date(yy, mm, dd)
        except Exception:
            pass

    m = DATE_DDMMYYYY_DOT_RE.search(t)
    if m:
        dd, mm, yy = map(int, m.groups())
        try:
            return date(yy, mm, dd)
        except Exception:
            pass

    return None


def parse_row_text_basic(txt: str) -> RowInfo:
    txt = " ".join((txt or "").split())
    played_on = extract_date_any(txt)

    m_id = ID_RE.search(txt)
    report_id = m_id.group(1) if m_id else None
    return RowInfo(report_id=report_id, played_on=played_on, raw_text=txt)


def in_range(d: Optional[date], d_from: date, d_to: date) -> bool:
    if not d:
        return True  # allow unknown dates; will still download (wide ranges work)
    return d_from <= d <= d_to


def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-. ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:180] if len(s) > 180 else s


def load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return set(map(str, data))
    except Exception:
        pass
    return set()


def save_state(path: Path, ids: set[str]) -> None:
    path.write_text(json.dumps(sorted(ids), ensure_ascii=False, indent=2), encoding="utf-8")


def unique_path(out_dir: Path, filename: str) -> Path:
    p = out_dir / filename
    if not p.exists():
        return p
    stem, ext = p.stem, p.suffix
    for n in range(1, 1000):
        cand = out_dir / f"{stem} ({n}){ext}"
        if not cand.exists():
            return cand
    raise RuntimeError("Cannot pick unique filename")


def extract_time_tag_any(txt: str) -> str:
    t = " ".join((txt or "").split())
    m = ROW_TIME_HHMMSS_RE.search(t)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m = ROW_TIME_HHMM_RE.search(t)
    if m:
        return f"{m.group(1)}{m.group(2)}00"
    m = ROW_TIME_6DIG_RE.search(t)
    if m:
        return m.group(1)
    return "000000"


def extract_fp_sp_names(row_txt: str) -> tuple[Optional[str], Optional[str]]:
    t = " ".join((row_txt or "").split())

    m = re.search(r"\bFP\b[\s:_-]*([^|]+?)\s+\bSP\b[\s:_-]*([^|]+)", t, re.IGNORECASE)
    if m:
        fp = m.group(1).strip(" _-|")
        sp = m.group(2).strip(" _-|")
        return fp or None, sp or None

    m = re.search(r"\bSP\b[\s:_-]*([^|]+?)\s+\bFP\b[\s:_-]*([^|]+)", t, re.IGNORECASE)
    if m:
        sp = m.group(1).strip(" _-|")
        fp = m.group(2).strip(" _-|")
        return fp or None, sp or None

    return None, None


def try_accept_cookies(page) -> None:
    for label in ("Accept", "I agree", "Agree", "OK"):
        try:
            btn = page.get_by_role("button", name=re.compile(rf"^{label}$", re.I))
            if btn.count() and btn.first.is_visible():
                btn.first.click(timeout=2000)
                page.wait_for_timeout(300)
                return
        except Exception:
            pass


def closest_row_text(link) -> str:
    for xp in ("xpath=ancestor::tr[1]", "xpath=ancestor::*[@role='row'][1]"):
        try:
            row = link.locator(xp)
            if row.count():
                t = row.first.inner_text(timeout=3000)
                if t and t.strip():
                    return t
        except Exception:
            pass
    try:
        return link.inner_text(timeout=2000)
    except Exception:
        return ""


def click_next_page(page) -> bool:
    # rel=next
    try:
        a = page.locator("ul.pagination a[rel='next'], nav ul.pagination a[rel='next']")
        if a.count() and a.first.is_visible():
            a.first.click()
            page.wait_for_load_state("domcontentloaded")
            return True
    except Exception:
        pass

    # aria-label Next
    try:
        a = page.locator("ul.pagination a[aria-label='Next'], nav ul.pagination a[aria-label='Next']")
        if a.count() and a.first.is_visible():
            a.first.click()
            page.wait_for_load_state("domcontentloaded")
            return True
    except Exception:
        pass

    # arrow / Next text
    for rx in (r"^(›|»|>)$", r"^(Next|Older)$"):
        try:
            a = page.locator("ul.pagination a, nav ul.pagination a", has_text=re.compile(rx, re.I))
            if a.count() and a.first.is_visible():
                a.first.click()
                page.wait_for_load_state("domcontentloaded")
                return True
        except Exception:
            pass

    return False


def _wait_reports_ready(page) -> None:
    page.wait_for_load_state("domcontentloaded")
    # try multiple selectors; do not hard-fail
    for sel in (
        "a:has-text('Download Report')",
        "button:has-text('Download Report')",
        "text=Download Report",
    ):
        try:
            page.wait_for_selector(sel, timeout=15000)
            return
        except Exception:
            continue


def _get_download_links(page):
    links = page.locator("a:has-text('Download Report')")
    try:
        if links.count() == 0:
            links = page.locator("button:has-text('Download Report')")
    except Exception:
        links = page.locator("a:has-text('Download Report')")
    return links


def download_reports(
    date_from: date,
    date_to: date,
    user_filter: str,
    out_dir: Path,
    progress_cb,
    stop_event: threading.Event,
):
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
    except Exception:
        raise RuntimeError("Playwright not installed: pip install playwright && playwright install chromium")

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "downloaded_ids.json"
    downloaded_ids = load_state(state_path)

    user_filter_norm = (user_filter or "").strip().lower()

    def log(msg: str):
        progress_cb(msg)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        log(f"Open: {URL}")
        page.goto(URL, wait_until="networkidle", timeout=60000)
        try_accept_cookies(page)
        _wait_reports_ready(page)

        processed_links = set()
        downloaded = 0
        matched = 0
        scanned = 0

        max_pages = 5000

        for page_idx in range(1, max_pages + 1):
            if stop_event.is_set():
                log("Stopped")
                break

            _wait_reports_ready(page)
            links = _get_download_links(page)

            try:
                count = links.count()
            except Exception:
                count = 0

            if count == 0:
                if page_idx == 1:
                    log("No 'Download Report' links found (page structure changed or blocked).")
                if not click_next_page(page):
                    break
                continue

            page_dates = []

            for i in range(count):
                if stop_event.is_set():
                    break

                link = links.nth(i)

                row_txt = closest_row_text(link)
                info = parse_row_text_basic(row_txt)

                scanned += 1
                if info.played_on:
                    page_dates.append(info.played_on)

                try:
                    href = link.get_attribute("href")
                except Exception:
                    href = None

                sig = href or f"p{page_idx}:i{i}:{info.report_id or ''}"
                if sig in processed_links:
                    continue
                processed_links.add(sig)

                if info.report_id and info.report_id in downloaded_ids:
                    continue
                if not in_range(info.played_on, date_from, date_to):
                    continue
                if user_filter_norm and user_filter_norm not in (info.raw_text or "").lower():
                    continue

                matched += 1

                try:
                    log(f"Download: {info.played_on or 'unknown date'} id={info.report_id or 'noid'}")
                    with page.expect_download(timeout=45000) as dl_info:
                        link.click()
                    dl = dl_info.value

                    suggested = dl.suggested_filename or "report.log"
                    ext = Path(suggested).suffix or ".log"

                    # if row has no date, try from suggested filename
                    played_on = info.played_on or extract_date_any(suggested) or extract_date_any(info.raw_text)

                    date_tag = played_on.strftime("%Y-%m-%d") if played_on else "nodate"
                    time_tag = extract_time_tag_any(info.raw_text + " " + suggested)

                    fp_name, sp_name = extract_fp_sp_names(info.raw_text)
                    if fp_name or sp_name:
                        fp_name = fp_name or "unknown"
                        sp_name = sp_name or "unknown"
                        fname = safe_name(f"{date_tag}_{time_tag}_FP_{fp_name}_SP_{sp_name}{ext}")
                    else:
                        # fallback: keep suggested name, but still try to make it stable
                        fname = safe_name(Path(suggested).name)
                        if not Path(fname).suffix:
                            fname += ext

                    target = unique_path(out_dir, fname)
                    dl.save_as(str(target))

                    if info.report_id:
                        downloaded_ids.add(info.report_id)
                        save_state(state_path, downloaded_ids)

                    downloaded += 1
                    log(f"Saved: {target.name}")

                except PWTimeoutError:
                    log("Download failed: timeout")
                except Exception as e:
                    log(f"Download error: {e}")

            # early stop if the table is sorted by newest first and we passed date_from
            if page_dates and max(page_dates) < date_from:
                break

            if not click_next_page(page):
                break

        save_state(state_path, downloaded_ids)
        context.close()
        browser.close()
        log(f"Done. Matched: {matched} | Downloaded: {downloaded} | Folder: {out_dir}")


# -----------------------------
# Tkinter GUI (single page; output left)
# -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WotR Tool")
        self.geometry("1400x850")
        self.minsize(1200, 700)

        self._q = queue.Queue()

        self._dl_thread = None
        self._dl_stop = threading.Event()

        self._ov_thread = None
        self._ov_stop = threading.Event()

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        root.columnconfigure(0, weight=4)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        out_frame = ttk.LabelFrame(root, text="Output")
        out_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        out_frame.rowconfigure(0, weight=1)
        out_frame.columnconfigure(0, weight=1)

        self.out_txt = tk.Text(out_frame, wrap="word")
        sb = ttk.Scrollbar(out_frame, orient="vertical", command=self.out_txt.yview)
        self.out_txt.configure(yscrollcommand=sb.set)
        self.out_txt.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        controls = ttk.Frame(root)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        ttk.Button(controls, text="Clear output", command=self._clear_output).grid(
            row=0, column=0, sticky="we", pady=(0, 10)
        )

        # Log statistics
        lf1 = ttk.LabelFrame(controls, text="Log statistics")
        lf1.grid(row=1, column=0, sticky="we", pady=(0, 10))
        lf1.columnconfigure(0, weight=1)

        self.local_path_var = tk.StringVar()
        self.local_only_player_var = tk.StringVar(value="")

        ttk.Label(lf1, text="Path").grid(row=0, column=0, sticky="w")
        ttk.Entry(lf1, textvariable=self.local_path_var).grid(row=1, column=0, sticky="we", pady=(0, 6))
        ttk.Button(lf1, text="Choose file", command=self._pick_local_file).grid(row=2, column=0, sticky="we")
        ttk.Button(lf1, text="Choose folder", command=self._pick_local_folder).grid(row=3, column=0, sticky="we", pady=(6, 6))

        ttk.Label(lf1, text="Only player").grid(row=4, column=0, sticky="w")
        ttk.Entry(lf1, textvariable=self.local_only_player_var).grid(row=5, column=0, sticky="we", pady=(0, 6))
        ttk.Button(lf1, text="Analyze", command=self._analyze_local).grid(row=6, column=0, sticky="we")

        # Download reports
        lf2 = ttk.LabelFrame(controls, text="Download reports")
        lf2.grid(row=2, column=0, sticky="we", pady=(0, 10))
        lf2.columnconfigure(0, weight=1)

        self.dl_from_var = tk.StringVar(value="2025-01-01")
        self.dl_to_var = tk.StringVar(value="2026-12-31")
        self.dl_user_var = tk.StringVar(value="")
        self.dl_out_var = tk.StringVar(value=str(Path.cwd() / "site_logs"))

        ttk.Label(lf2, text="From (YYYY-MM-DD)").grid(row=0, column=0, sticky="w")
        ttk.Entry(lf2, textvariable=self.dl_from_var).grid(row=1, column=0, sticky="we", pady=(0, 6))

        ttk.Label(lf2, text="To (YYYY-MM-DD)").grid(row=2, column=0, sticky="w")
        ttk.Entry(lf2, textvariable=self.dl_to_var).grid(row=3, column=0, sticky="we", pady=(0, 6))

        ttk.Label(lf2, text="User filter").grid(row=4, column=0, sticky="w")
        ttk.Entry(lf2, textvariable=self.dl_user_var).grid(row=5, column=0, sticky="we", pady=(0, 6))

        ttk.Label(lf2, text="Output folder").grid(row=6, column=0, sticky="w")
        ttk.Entry(lf2, textvariable=self.dl_out_var).grid(row=7, column=0, sticky="we", pady=(0, 6))
        ttk.Button(lf2, text="Choose", command=self._pick_dl_out).grid(row=8, column=0, sticky="we", pady=(0, 6))

        self.btn_dl_start = ttk.Button(lf2, text="Start", command=self._start_download)
        self.btn_dl_stop = ttk.Button(lf2, text="Stop", command=self._stop_download, state="disabled")
        self.btn_dl_start.grid(row=9, column=0, sticky="we")
        self.btn_dl_stop.grid(row=10, column=0, sticky="we", pady=(6, 0))

        # Overall stats
        lf3 = ttk.LabelFrame(controls, text="Overall statistics (downloaded site logs)")
        lf3.grid(row=3, column=0, sticky="we")
        lf3.columnconfigure(0, weight=1)

        self.ov_folder_var = tk.StringVar(value=str(Path.cwd() / "site_logs"))
        self.ov_dedupe_var = tk.BooleanVar(value=True)

        ttk.Label(lf3, text="Folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(lf3, textvariable=self.ov_folder_var).grid(row=1, column=0, sticky="we", pady=(0, 6))
        ttk.Button(lf3, text="Choose", command=self._pick_ov_folder).grid(row=2, column=0, sticky="we", pady=(0, 6))

        ttk.Checkbutton(lf3, text="Deduplicate games", variable=self.ov_dedupe_var).grid(
            row=3, column=0, sticky="w", pady=(0, 6)
        )

        self.btn_ov = ttk.Button(lf3, text="Compute", command=self._start_overall)
        self.btn_ov_stop = ttk.Button(lf3, text="Stop", command=self._stop_overall, state="disabled")
        self.btn_ov.grid(row=4, column=0, sticky="we")
        self.btn_ov_stop.grid(row=5, column=0, sticky="we", pady=(6, 0))

    def _clear_output(self):
        self.out_txt.delete("1.0", "end")

    def _append_out(self, s: str):
        self.out_txt.insert("end", (s if s.endswith("\n") else s + "\n"))
        self.out_txt.see("end")

    def _pick_local_file(self):
        path = filedialog.askopenfilename(
            title="Choose log file",
            filetypes=[("Log files", "*.log *.txt"), ("All files", "*.*")],
        )
        if path:
            self.local_path_var.set(path)

    def _pick_local_folder(self):
        p = filedialog.askdirectory(title="Choose folder with logs")
        if p:
            self.local_path_var.set(p)

    def _analyze_local(self):
        src = self.local_path_var.get().strip()
        if not src:
            messagebox.showerror("Error", "Select a file or folder")
            return
        path = Path(src)
        if not path.exists():
            messagebox.showerror("Error", "Path not found")
            return

        only_player = self.local_only_player_var.get().strip()

        try:
            if path.is_file():
                info = analyze_single_log(path, full_file=False)
                report = build_text_report(info.counts, only_player=only_player)
            else:
                picks = pick_best_logs_from_folder(path)
                total_counts = merge_counts(picks)
                report = build_text_report(total_counts, only_player=only_player)

            self._append_out(report)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _pick_dl_out(self):
        p = filedialog.askdirectory(title="Choose output folder")
        if p:
            self.dl_out_var.set(p)

    def _start_download(self):
        if self._dl_thread and self._dl_thread.is_alive():
            return

        try:
            d_from = datetime.strptime(self.dl_from_var.get().strip(), "%Y-%m-%d").date()
            d_to = datetime.strptime(self.dl_to_var.get().strip(), "%Y-%m-%d").date()
        except Exception:
            messagebox.showerror("Error", "Date format: YYYY-MM-DD")
            return

        out_dir = Path(self.dl_out_var.get().strip())
        user_filter = self.dl_user_var.get().strip()

        self._dl_stop.clear()
        self.btn_dl_start.configure(state="disabled")
        self.btn_dl_stop.configure(state="normal")

        def progress(msg: str):
            self._q.put(("dl_log", msg))

        def run():
            try:
                download_reports(
                    date_from=d_from,
                    date_to=d_to,
                    user_filter=user_filter,
                    out_dir=out_dir,
                    progress_cb=progress,
                    stop_event=self._dl_stop,
                )
            except Exception as e:
                self._q.put(("dl_log", f"Error: {e}"))
            finally:
                self._q.put(("dl_done", None))

        self._dl_thread = threading.Thread(target=run, daemon=True)
        self._dl_thread.start()

    def _stop_download(self):
        self._dl_stop.set()

    def _pick_ov_folder(self):
        p = filedialog.askdirectory(title="Choose folder with downloaded logs")
        if p:
            self.ov_folder_var.set(p)

    def _start_overall(self):
        if self._ov_thread and self._ov_thread.is_alive():
            return

        folder = Path(self.ov_folder_var.get().strip())
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Error", "Folder not found")
            return

        dedupe = bool(self.ov_dedupe_var.get())

        self._ov_stop.clear()
        self.btn_ov.configure(state="disabled")
        self.btn_ov_stop.configure(state="normal")

        def progress(msg: str):
            self._q.put(("ov_log", msg))

        def run():
            try:
                report = overall_stats_from_site_folder(folder, dedupe, progress, self._ov_stop)
                self._q.put(("ov_out", report))
            except Exception as e:
                self._q.put(("ov_out", f"Error: {e}\n"))
            finally:
                self._q.put(("ov_done", None))

        self._ov_thread = threading.Thread(target=run, daemon=True)
        self._ov_thread.start()

    def _stop_overall(self):
        self._ov_stop.set()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "dl_log":
                    self._append_out(str(payload))
                elif kind == "dl_done":
                    self.btn_dl_start.configure(state="normal")
                    self.btn_dl_stop.configure(state="disabled")
                elif kind == "ov_log":
                    self._append_out(str(payload))
                elif kind == "ov_out":
                    self._append_out(str(payload))
                elif kind == "ov_done":
                    self.btn_ov.configure(state="normal")
                    self.btn_ov_stop.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


if __name__ == "__main__":
    App().mainloop()
