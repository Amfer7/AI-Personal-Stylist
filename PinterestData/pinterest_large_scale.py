"""
Pinterest Fashion Dataset Builder — Large Scale (50K+ images)
==============================================================
Designed for unattended overnight runs. Key features:
  - Resumable: tracks progress in a SQLite DB, safe to Ctrl+C and restart
  - Parallel: N worker threads download concurrently
  - Rate-limited: per-worker delays + exponential backoff on failure
  - Deduplicated: image hash checked before saving
  - 60+ curated fashion queries across styles, genders, seasons, aesthetics

Requirements:
    pip install gallery-dl Pillow tqdm pandas requests

Usage:
    python pinterest_large_scale.py scrape   --workers 4 --max-per-query 1000
    python pinterest_large_scale.py build
    python pinterest_large_scale.py stats
"""

import os
import json
import csv
import time
import shutil
import hashlib
import sqlite3
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from typing import Optional
import concurrent.futures

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────
# 60+ FASHION QUERIES  (broad + niche = diverse date distribution)
# ──────────────────────────────────────────────────────────────────

FASHION_QUERIES = [
    # Core outfit styles
    "outfit of the day women",
    "outfit of the day men",
    "street style fashion",
    "casual chic outfit women",
    "business casual outfit women",
    "business casual outfit men",
    "smart casual men outfit",
    "date night outfit ideas",
    "brunch outfit ideas",
    "summer outfit inspo",
    "winter outfit layering",
    "autumn fall fashion outfit",
    "spring fashion outfit ideas",

    # Aesthetics
    "minimalist outfit aesthetic",
    "y2k fashion aesthetic outfit",
    "cottagecore outfit ideas",
    "dark academia outfit",
    "old money aesthetic outfit",
    "clean girl aesthetic outfit",
    "coastal grandmother style",
    "boho chic outfit",
    "streetwear outfit men",
    "streetwear outfit women",
    "edgy fashion outfit",
    "preppy outfit ideas",
    "soft girl outfit aesthetic",
    "grunge fashion outfit",
    "indie aesthetic outfit",

    # Specific garment combos
    "jeans and blazer outfit",
    "midi skirt outfit",
    "trench coat outfit ideas",
    "leather jacket outfit",
    "oversized blazer outfit",
    "wide leg pants outfit",
    "linen outfit summer",
    "knit set outfit",
    "cargo pants outfit",
    "maxi dress outfit",
    "co ord set outfit",
    "turtleneck outfit winter",
    "loafers outfit women",

    # Occasion-based
    "wedding guest outfit",
    "festival fashion outfit",
    "vacation beach outfit",
    "airport outfit ideas",
    "graduation outfit ideas",
    "concert outfit women",
    "gym workout outfit women",
    "athleisure outfit",

    # Color & pattern themed
    "monochrome outfit ideas",
    "all black outfit",
    "neutral tones outfit",
    "pastel outfit aesthetic",
    "floral print outfit",
    "striped outfit ideas",

    # Current trend-tagged (good for temporal spread)
    "fashion trends 2023 outfit",
    "fashion trends 2024 outfit",
    "fashion inspo 2024",
    "trending outfit ideas",
    "viral outfit tiktok fashion",
    "pinterest fashion 2024",
    "instagram fashion outfit",

    # International / runway-inspired
    "paris fashion week street style",
    "new york fashion week street style",
    "tokyo street style fashion",
    "milan fashion week looks",
    "copenhagen street style",

    # New micro-trend aesthetics (none of these existed in original list)
    "quiet luxury outfit",
    "mob wife aesthetic outfit",
    "coquette aesthetic outfit",
    "gorpcore outfit",
    "balletcore outfit",
    "office siren outfit",
    "dopamine dressing outfit",
    "scandi minimalist outfit",
    "indie sleaze outfit",

    # New garment/trend combos not in original list
    "low rise pants outfit 2026",
    "sheer layering outfit",
    "platform loafers outfit",
    "balletcore flats outfit",
    "denim on denim outfit 2026",

    # New occasion type not covered
    "vacation resort outfit",

    # New color/pattern not covered
    "animal print outfit 2026",
    "color blocking outfit",

    # New year-tagged trend queries (2025/2026, vs original's 2023/2024)
    "fashion trends 2026 outfit",
    "fashion inspo 2026",
    "instagram baddie outfit 2026",

    # New runway city not covered (original had Paris/NY/Tokyo/Milan/Copenhagen)
    "seoul street style fashion",

    #INDIAN FASHION QUERIES (added for diversity and temporal spread)
    # Ethnic wear — core
    "indian ethnic wear outfit",
    "saree draping styles 2026",
    "designer saree new collection",
    "lehenga choli outfit",
    "anarkali suit outfit",
    "salwar kameez outfit",
    "kurta set men",
    "kurti with jeans outfit",
    "palazzo suit outfit",
    "sharara suit outfit",

    # Indo-western / fusion
    "indo western outfit women",
    "indo western outfit men",
    "fusion wear saree",
    "crop top with lehenga",
    "dhoti pants outfit indian",
    "jacket lehenga outfit",

    # Occasion-based (Indian)
    "indian wedding guest outfit",
    "mehendi outfit ideas",
    "sangeet outfit ideas",
    "haldi outfit ideas",
    "diwali outfit ideas 2026",
    "navratri outfit ideas",
    "indian festival outfit",
    "reception outfit indian bride",
    "engagement outfit indian",
    "indian wedding groom outfit",

    # Regional / traditional styles
    "banarasi saree",
    "south indian silk saree",
    "punjabi suit outfit",
    "rajasthani traditional outfit",
    "bengali saree style",
    "kanjivaram saree",

    # Contemporary Indian fashion scene
    "lakme fashion week 2026",
    "india couture week looks",
    "bollywood celebrity airport look",
    "indian influencer fashion 2026",
    "modern indian ethnic wear 2026",
    "indian street style fashion",
    "sustainable indian fashion brands",
    "indian designer wear 2026",

    # Everyday / casual Indian fashion
    "indian casual kurti outfit",
    "indian office wear women",
    "co ord set indian ethnic",
    "indowestern party wear",
]


RESIZE_TO  = (224, 224)
MIN_SIZE   = (100, 100)

# ──────────────────────────────────────────────────────────────────
# SQLITE PROGRESS TRACKER  (enables resumability)
# ──────────────────────────────────────────────────────────────────

class ProgressDB:
    """
    Thread-safe SQLite tracker.
    Remembers which queries are done and which image hashes exist,
    so a restart skips already-completed work.
    """

    def __init__(self, db_path: str):
        self.path = db_path
        self.lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        return sqlite3.connect(self.path, check_same_thread=False)

    def _init(self):
        with self.lock, self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    query TEXT PRIMARY KEY,
                    status TEXT,        -- 'done' | 'failed'
                    pin_count INTEGER,
                    finished_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hashes (
                    hash TEXT PRIMARY KEY
                )
            """)
            conn.commit()

    def query_done(self, query: str) -> bool:
        with self.lock, self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM queries WHERE query=?", (query,)
            ).fetchone()
            return row is not None and row[0] == "done"

    def mark_query(self, query: str, status: str, pin_count: int = 0):
        with self.lock, self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO queries (query, status, pin_count, finished_at)
                VALUES (?, ?, ?, ?)
            """, (query, status, pin_count, datetime.utcnow().isoformat()))
            conn.commit()

    def hash_seen(self, h: str) -> bool:
        with self.lock, self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM hashes WHERE hash=?", (h,)
            ).fetchone() is not None

    def add_hash(self, h: str):
        with self.lock, self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO hashes (hash) VALUES (?)", (h,)
            )
            conn.commit()

    def summary(self) -> dict:
        with self.lock, self._conn() as conn:
            done  = conn.execute("SELECT COUNT(*) FROM queries WHERE status='done'").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            imgs  = conn.execute("SELECT COUNT(*) FROM hashes").fetchone()[0]
            return {"queries_done": done, "queries_total": total, "unique_images": imgs}


# ──────────────────────────────────────────────────────────────────
# GALLERY-DL CONFIG
# ──────────────────────────────────────────────────────────────────

def build_gallerydl_config(output_dir: str, cookies_file: Optional[str] = None,
                           username: Optional[str] = None, password: Optional[str] = None) -> str:
    config = {
        "extractor": {
            "pinterest": {
                "videos":   False,
                "metadata": True,
            },
            "base-directory": str(Path(output_dir) / "raw"),
        },
        "downloader": {
            "retries":  4,
            "timeout":  45,
            "rate":     "800k",
        },
        "output": {
            "log": {"level": "warning"},
        },
    }
    if cookies_file:
        config["extractor"]["pinterest"]["cookies"] = cookies_file
    if username:
        config["extractor"]["pinterest"]["username"] = username
    if password:
        config["extractor"]["pinterest"]["password"] = password

    config_path = Path(output_dir) / "gdl_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return str(config_path)


# ──────────────────────────────────────────────────────────────────
# SINGLE-QUERY SCRAPER  (called by worker threads)
# ──────────────────────────────────────────────────────────────────

def scrape_query(query: str, output_dir: str, config_path: str,
                 max_pins: int, retries: int = 2) -> int:
    """
    Run gallery-dl for one query. Returns number of files downloaded.
    Retries with exponential backoff on failure.
    """
    url = f"https://in.pinterest.com/search/pins/?q={query.replace(' ', '+')}"
    cmd = [
        "gallery-dl",
        "--config",        config_path,
        "--range",         f"1-{max_pins}",
        "--write-metadata",
        "--no-mtime",
        url,
    ]

    for attempt in range(retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            raw_path = Path(output_dir) / "raw"
            return sum(1 for _ in raw_path.rglob("*.jpg")) + \
                   sum(1 for _ in raw_path.rglob("*.png")) + \
                   sum(1 for _ in raw_path.rglob("*.webp")) + \
                   sum(1 for _ in raw_path.rglob("*.heic"))
        if attempt < retries:
            wait = 10 * (2 ** attempt)
            time.sleep(wait)

    return 0


# ──────────────────────────────────────────────────────────────────
# PARALLEL SCRAPE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────

def run_parallel_scrape(output_dir: str, max_per_query: int,
                        workers: int, cookies: Optional[str] = None,
                        username: Optional[str] = None, password: Optional[str] = None,
                        queries: Optional[list] = None):
    db          = ProgressDB(str(Path(output_dir) / "progress.db"))
    config_path = build_gallerydl_config(output_dir, cookies, username, password)
    query_list  = queries or FASHION_QUERIES

    # Filter out already-completed queries
    pending = [q for q in query_list if not db.query_done(q)]
    print(f"\n[scrape] {len(pending)} queries to run | {len(query_list) - len(pending)} already done")
    print(f"[scrape] workers={workers}  max_per_query={max_per_query}")
    print(f"[scrape] estimated images: ~{len(pending) * max_per_query} (before dedup)")

    if not pending:
        print("[scrape] Nothing to do. Run `build` next.")
        return

    pbar = tqdm(total=len(pending), desc="Queries", unit="query")
    lock = threading.Lock()

    def worker(query: str):
        # Each worker adds a small random jitter to avoid synchronized requests
        time.sleep(threading.current_thread().ident % 5)
        try:
            count = scrape_query(query, output_dir, config_path, max_per_query)
            db.mark_query(query, "done", count)
        except Exception as e:
            db.mark_query(query, "failed")
        finally:
            with lock:
                stats = db.summary()
                pbar.set_postfix({"imgs": stats["unique_images"]})
                pbar.update(1)
            # Polite pause between queries per worker
            time.sleep(3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pool.map(worker, pending)

    pbar.close()
    stats = db.summary()
    print(f"\n[scrape done] queries: {stats['queries_done']}/{len(query_list)}")
    print(f"[scrape done] Run `build` to process images into the clean dataset.")


# ──────────────────────────────────────────────────────────────────
# METADATA EXTRACTION
# ──────────────────────────────────────────────────────────────────

def parse_pin_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    # Try parsing with timezone intact first
    for fmt in ["%a, %d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"]:
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m")
        except ValueError:
            continue
    # Fallback: strip timezone suffix and retry
    clean = raw.split("+")[0].split("Z")[0].strip()
    for fmt in ["%a, %d %b %Y %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(clean, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return raw[:7] if len(raw) >= 7 and raw[4] == "-" else None


def read_sidecar(json_path: Path) -> dict:
    try:
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {}

    # Board name
    board = d.get("board", "")
    if isinstance(board, dict):
        board = board.get("name", "")

    # Date — gallery-dl writes "created_at" with RFC format
    raw_date = d.get("created_at") or d.get("date") or ""

    # Pin ID
    pin_id = d.get("id") or d.get("pin_id") or ""

    # Description — fall back to visual annotations if empty
    description = d.get("description", "").strip()
    if not description:
        annotations = d.get("pin_join", {}).get("visual_annotation", [])
        description = ", ".join(annotations[:8])

    # Query — gallery-dl writes "search" field (with + separators)
    query = d.get("search", "").replace("+", " ")

    return {
        "pin_id":      str(pin_id),
        "description": description[:300],
        "board":       board,
        "date_raw":    raw_date,
        "date_ym":     parse_pin_date(raw_date),
        "query":       query,
    }


# ──────────────────────────────────────────────────────────────────
# IMAGE PROCESSOR
# ──────────────────────────────────────────────────────────────────

def process_image(src: Path, dest: Path) -> Optional[tuple]:
    try:
        img = Image.open(src).convert("RGB")
        w, h = img.size
        if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
            return None
        img.resize(RESIZE_TO, Image.LANCZOS).save(dest, "JPEG", quality=90)
        return (w, h)
    except Exception:
        return None


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ──────────────────────────────────────────────────────────────────

def build_dataset(raw_dir: str, dataset_dir: str):
    raw_path     = Path(raw_dir)
    ds_path      = Path(dataset_dir)
    images_path  = ds_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_imgs = [p for p in raw_path.rglob("*") if p.suffix.lower() in exts]
    print(f"\n[build] {len(all_imgs)} raw images found")

    records     = []
    seen_hashes = set()

    for img_path in tqdm(all_imgs, desc="Processing", unit="img"):
        # Sidecar
        sidecar = img_path.with_suffix(img_path.suffix + ".json")
        if not sidecar.exists():
            sidecar = img_path.with_suffix(".json")
        meta = read_sidecar(sidecar) if sidecar.exists() else {}

        # Dedup
        h = file_hash(img_path)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Use metadata pin_id if available, otherwise use the filename gallery-dl gave it
        pin_id    = meta.get("pin_id") or img_path.stem
        dest      = images_path / f"{pin_id}.jpg"
        dims      = process_image(img_path, dest)
        if dims is None:
            continue

        records.append({
            "image_path":  str(dest.relative_to(ds_path)),
            "pin_id":      pin_id,
            "date_ym":     meta.get("date_ym"),
            "date_raw":    meta.get("date_raw", ""),
            "board":       meta.get("board", ""),
            "query":       meta.get("query", ""),
            "description": meta.get("description", ""),
            "orig_width":  dims[0],
            "orig_height": dims[1],
            "hash":        h,
        })

    df = pd.DataFrame(records)
    csv_out = ds_path / "metadata.csv"
    df.to_csv(csv_out, index=False)

    dated = df["date_ym"].notna().sum()
    print(f"\n[build done]")
    print(f"  Unique images:      {len(df)}")
    print(f"  With valid date:    {dated} ({100*dated//max(1,len(df))}%)")
    print(f"  Date range:         {df['date_ym'].dropna().min()} → {df['date_ym'].dropna().max()}")
    print(f"  CSV saved:          {csv_out}")
    return df


# ──────────────────────────────────────────────────────────────────
# STATS
# ──────────────────────────────────────────────────────────────────

def print_stats(dataset_dir: str):
    df = pd.read_csv(Path(dataset_dir) / "metadata.csv")
    print(f"\n{'='*42}")
    print(f"  Total images:   {len(df)}")
    print(f"  Dated images:   {df['date_ym'].notna().sum()}")
    print(f"  Unique queries: {df['query'].nunique()}")

    if df["date_ym"].notna().any():
        monthly = df.groupby("date_ym").size().sort_index().tail(30)
        max_c   = monthly.max()
        print(f"\n  Monthly distribution (last 30 months):")
        for ym, c in monthly.items():
            bar = "█" * max(1, int(30 * c / max_c))
            print(f"    {ym}  {bar:<31} {c}")

    top_q = df.groupby("query").size().sort_values(ascending=False).head(10)
    print(f"\n  Top queries by image count:")
    for q, c in top_q.items():
        print(f"    {c:>5}  {q}")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Large-scale Pinterest fashion dataset builder"
    )
    subs = parser.add_subparsers(dest="cmd")

    sp = subs.add_parser("scrape")
    sp.add_argument("--output-dir",     default="./pinterest_raw")
    sp.add_argument("--max-per-query",  type=int, default=1000,
                    help="Pins per query. 1000 × 60 queries = up to 60K images before dedup.")
    sp.add_argument("--workers",        type=int, default=3,
                    help="Parallel download threads. 3-4 is safe; >6 risks IP blocks.")
    sp.add_argument("--cookies",        default=None,
                    help="Path to cookies.txt (optional).")
    sp.add_argument("--username",       default=None,
                    help="Pinterest email address.")
    sp.add_argument("--password",       default=None,
                    help="Pinterest password.")
    sp.add_argument("--queries",        nargs="*",
                    help="Override query list. Defaults to 60+ built-in fashion queries.")

    bp = subs.add_parser("build")
    bp.add_argument("--raw-dir",       default="./pinterest_raw/raw")
    bp.add_argument("--dataset-dir",   default="./pinterest_dataset")

    st = subs.add_parser("stats")
    st.add_argument("--dataset-dir",   default="./pinterest_dataset")

    args = parser.parse_args()

    if args.cmd == "scrape":
        run_parallel_scrape(
            output_dir    = args.output_dir,
            max_per_query = args.max_per_query,
            workers       = args.workers,
            cookies       = args.cookies,
            username      = args.username,
            password      = args.password,
            queries       = args.queries or None,
        )
    elif args.cmd == "build":
        build_dataset(args.raw_dir, args.dataset_dir)
    elif args.cmd == "stats":
        print_stats(args.dataset_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()