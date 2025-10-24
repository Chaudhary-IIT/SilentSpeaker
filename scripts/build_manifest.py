# scripts/build_manifest.py
import csv, re
from pathlib import Path

VIDEO_EXTS = {".mpg", ".mp4", ".avi", ".mov", ".mkv"}
ALIGN_EXTS = {".align", ".txt", ".lab"}
DROP = {"sil", "sp", "spn"}  # silence-like tokens

ALLOW = set(" abcdefghijklmnopqrstuvwxyz'")

def clean_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^ a-z']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def parse_align_grid(path: Path) -> str:
    """
    GRID-style lines are usually: <start> <end> <word>
    Fallback: if word-first lines are found, handle those too.
    """
    toks = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) >= 3:
                # If first two are numbers, take the last item as the word
                if parts[0].replace(".", "", 1).isdigit() and parts[1].replace(".", "", 1).isdigit():
                    w = parts[-1]
                else:
                    # word-first fallback
                    w = parts[0]
            else:
                w = parts[-1]
            wl = w.lower()
            if wl in DROP:
                continue
            toks.append(wl)
    txt = " ".join(toks)
    txt = "".join(ch for ch in txt if ch in ALLOW)
    return clean_text(txt)

def find_align_for(video: Path, root: Path) -> Path | None:
    """
    Match sX/<stem>.<ext> to alignments/sX/<stem>.<align_ext>
    Also check same-dir or parent/alignments for robustness.
    """
    stem, spk = video.stem, video.parent.name  # e.g., spk = "s1"
    candidates = []
    # Standard GRID layout: <root>/alignments/<spk>/<stem>.align
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
    return candidates[0] if candidates else None

def build(root: str, outdir: str, val_frac: float = 0.1):
    root = Path(root).resolve()
    # Gather speakers like s1, s2, ...
    spk_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.lower().startswith("s")]
    vids = []
    for spk in spk_dirs:
        for p in spk.iterdir():
            if p.suffix.lower() in VIDEO_EXTS:
                vids.append(p)
    rows = []
    for v in sorted(vids):
        a = find_align_for(v, root)
        if not a:
            continue
        txt = parse_align_grid(a)
        if txt:
            rows.append((str(v.resolve()), txt))
    rows = list(dict.fromkeys(rows))
    if not rows:
        raise SystemExit("No video↔alignment pairs after parsing; confirm stems match and .align format under alignments/sX.")
    n = len(rows); n_val = max(1, int(n * val_frac))
    val = rows[-n_val:]; train = rows[:-n_val] if n > n_val else rows
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    for name, subset in [("train_manifest.csv", train), ("val_manifest.csv", val)]:
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["video_path","text"]); w.writerows(subset)
    print(f"Wrote {len(train)} train and {len(val)} val rows to {out}")
    
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()
    build(args.root, args.outdir, args.val_frac)
