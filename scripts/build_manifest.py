# scripts/build_manifest.py

import csv
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mpg", ".mp4", ".avi", ".mov", ".mkv"}
ALIGN_EXTS = {".align", ".txt", ".lab"}
DROP = {"sil", "sp", "spn"}  # silence-like tokens
ALLOW = set(" abcdefghijklmnopqrstuvwxyz'")


def clean_text(s: str) -> str:
    """Clean and normalize text."""
    s = s.lower().strip()
    s = re.sub(r"[^ a-z']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_align_grid(path: Path) -> str:
    """
    Parse GRID-style alignment files.
    
    GRID format:
        <start_time> <end_time> <word>
    
    Example:
        0.0 0.5 bin
        0.5 1.0 blue
    
    Fallback: if word-first lines are found, handle those too.
    """
    toks = []
    
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                
                parts = ln.split()
                if len(parts) >= 3:
                    # If first two are numbers, take the last item as the word
                    if (parts[0].replace(".", "", 1).replace("-", "", 1).isdigit() and 
                        parts[1].replace(".", "", 1).replace("-", "", 1).isdigit()):
                        w = parts[-1]
                    else:
                        # word-first fallback
                        w = parts[0]
                else:
                    w = parts[-1] if parts else ""
                
                wl = w.lower()
                if wl in DROP or not wl:
                    continue
                
                toks.append(wl)
        
        txt = " ".join(toks)
        txt = "".join(ch for ch in txt if ch in ALLOW)
        return clean_text(txt)
    
    except Exception as e:
        logger.warning(f"Failed to parse alignment file {path}: {e}")
        return ""


def find_align_for(video: Path, root: Path) -> Path | None:
    """
    Find alignment file for a video.
    
    Search strategy:
    1. Standard GRID layout: <root>/alignments/<speaker>/<video_stem>.align
    2. Same directory as video
    3. One level up: <root>/alignments/<video_stem>.align
    
    Args:
        video: Path to video file
        root: Root directory of dataset
    
    Returns:
        Path to alignment file or None if not found
    """
    stem, spk = video.stem, video.parent.name  # e.g., spk = "s1"
    candidates = []
    
    # Standard GRID layout: /alignments/sX/<stem>.align
    cand_dir = root / "alignments" / spk
    for ext in ALIGN_EXTS:
        c = cand_dir / f"{stem}{ext}"
        if c.exists():
            candidates.append(c)
    
    # Same directory
    for ext in ALIGN_EXTS:
        c = video.parent / f"{stem}{ext}"
        if c.exists():
            candidates.append(c)
    
    # One level up alignments
    for ext in ALIGN_EXTS:
        c = video.parent.parent / "alignments" / f"{stem}{ext}"
        if c.exists():
            candidates.append(c)
    
    if not candidates:
        logger.debug(f"No alignment found for {video.name}")
    
    return candidates[0] if candidates else None


def build(root: str, outdir: str, val_frac: float = 0.1, min_text_len: int = 1):
    """
    Build train/val manifest CSV files from video dataset.
    
    Args:
        root: Root directory containing speaker folders (s1, s2, ...)
        outdir: Output directory for manifest files
        val_frac: Fraction of data to use for validation (default: 0.1)
        min_text_len: Minimum text length to include (default: 1)
    
    Dataset structure expected:
        <root>/
            s1/
                video1.mpg
                video2.mpg
            s2/
                video1.mpg
            alignments/
                s1/
                    video1.align
                    video2.align
                s2/
                    video1.align
    
    Output:
        <outdir>/train_manifest.csv
        <outdir>/val_manifest.csv
    
    CSV format:
        video_path,text
        /path/to/video1.mp4,bin blue at a five now
        /path/to/video2.mp4,lay green in f zero please
    """
    root = Path(root).resolve()
    
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    
    logger.info(f"Building manifest from: {root}")
    
    # Gather speakers like s1, s2, ...
    spk_dirs = [
        d for d in root.iterdir() 
        if d.is_dir() and d.name.lower().startswith("s")
    ]
    
    if not spk_dirs:
        raise ValueError(f"No speaker directories (s1, s2, ...) found in {root}")
    
    logger.info(f"Found {len(spk_dirs)} speaker directories: {[d.name for d in spk_dirs]}")
    
    # Collect all videos
    vids = []
    for spk in spk_dirs:
        spk_vids = [p for p in spk.iterdir() if p.suffix.lower() in VIDEO_EXTS]
        logger.info(f"Speaker {spk.name}: {len(spk_vids)} videos")
        vids.extend(spk_vids)
    
    if not vids:
        raise ValueError(f"No video files found in speaker directories")
    
    logger.info(f"Total videos found: {len(vids)}")
    
    # Match videos with alignments
    rows = []
    skipped = 0
    
    for v in sorted(vids):
        a = find_align_for(v, root)
        if not a:
            skipped += 1
            continue
        
        txt = parse_align_grid(a)
        if txt and len(txt) >= min_text_len:
            rows.append((str(v.resolve()), txt))
        else:
            logger.debug(f"Skipped {v.name}: empty or too short text")
            skipped += 1
    
    # Remove duplicates while preserving order
    rows = list(dict.fromkeys(rows))
    
    logger.info(f"Processed: {len(rows)} valid pairs, {skipped} skipped")
    
    if not rows:
        raise SystemExit(
            "No video↔alignment pairs after parsing. "
            "Confirm:\n"
            "  1. Video stems match alignment file names\n"
            "  2. Alignment files are in .align format under alignments/sX\n"
            "  3. Alignment files contain valid text"
        )
    
    # Split train/val
    n = len(rows)
    n_val = max(1, int(n * val_frac))
    val = rows[-n_val:]
    train = rows[:-n_val] if n > n_val else rows
    
    logger.info(f"Split: {len(train)} train, {len(val)} val")
    
    # Write manifests
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    
    for name, subset in [("train_manifest.csv", train), ("val_manifest.csv", val)]:
        out_path = out / name
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_path", "text"])
            w.writerows(subset)
        
        logger.info(f"Wrote {len(subset)} rows to {out_path}")
        
        # Print sample rows
        if subset:
            logger.info(f"Sample from {name}:")
            for i, (path, text) in enumerate(subset[:3]):
                logger.info(f"  [{i+1}] {Path(path).name}: '{text}'")
    
    logger.info(f"✅ Manifest building complete!")
    logger.info(f"   Output directory: {out.resolve()}")


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Build train/val manifest CSV files from GRID-style video dataset"
    )
    ap.add_argument(
        "--root", 
        required=True,
        help="Root directory containing speaker folders (s1, s2, ...) and alignments/"
    )
    ap.add_argument(
        "--outdir", 
        default="data",
        help="Output directory for manifest files (default: data)"
    )
    ap.add_argument(
        "--val_frac", 
        type=float, 
        default=0.1,
        help="Validation fraction (default: 0.1)"
    )
    ap.add_argument(
        "--min_text_len",
        type=int,
        default=1,
        help="Minimum text length to include (default: 1)"
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = ap.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        build(args.root, args.outdir, args.val_frac, args.min_text_len)
    except Exception as e:
        logger.error(f"❌ Failed to build manifest: {e}")
        raise
