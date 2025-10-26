# scripts/train_vsr.py

# Usage:
# python -u scripts/train_vsr.py data/train_manifest.csv data/val_manifest.csv --device cuda

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from dataset_vsr import MouthROIVideoDataset, pad_collate
from model_vsr import CNN3D_BiGRU
from vocab import build_vocab, save_vocab

def _prepare_dataloaders(train_manifest, val_manifest, bs, use_cuda):
    ds_tr = MouthROIVideoDataset(train_manifest, char2idx=None)
    ds_va = MouthROIVideoDataset(val_manifest, char2idx=None)
    dl_kwargs = dict(batch_size=bs, num_workers=0, collate_fn=pad_collate, pin_memory=use_cuda)
    return ds_tr, ds_va, DataLoader(ds_tr, shuffle=True, **dl_kwargs), DataLoader(ds_va, shuffle=False, **dl_kwargs)

def _attach_vocab(ds_tr, ds_va):
    char2idx, idx2char = build_vocab()
    ds_tr.char2idx = char2idx
    ds_va.char2idx = char2idx
    return char2idx, idx2char

def _subset_flat_targets(yb, ylen, mask):
    """
    yb: 1D concatenated targets (sum_L,)
    ylen: (B,)
    mask: (B,) bool for samples to keep
    Returns yb_sub (1D), ylen_sub (B_sub,)
    """
    device = ylen.device
    ylen_cpu = ylen.detach().cpu()
    starts = torch.cumsum(torch.nn.functional.pad(ylen_cpu[:-1], (1,0)), dim=0) # (B,)
    idx_keep = torch.nonzero(mask, as_tuple=False).squeeze(1).cpu()
    segs = []
    lens = []
    for i in idx_keep.tolist():
        s = int(starts[i].item())
        e = s + int(ylen_cpu[i].item())
        segs.append(yb[s:e])
        lens.append(int(ylen_cpu[i].item()))
    if len(segs) == 0:
        return torch.zeros(0, dtype=yb.dtype, device=yb.device), torch.zeros(0, dtype=torch.long, device=device)
    yb_sub = torch.cat(segs, dim=0).to(yb.device)
    ylen_sub = torch.tensor(lens, dtype=torch.long, device=device)
    return yb_sub, ylen_sub

def train(train_manifest, val_manifest, out_dir="checkpoints", epochs=5, bs=4, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_vsr] device={device}")
    use_cuda = device.startswith("cuda")
    os.makedirs(out_dir, exist_ok=True)

    ds_tr, ds_va, dl_tr, dl_va = _prepare_dataloaders(train_manifest, val_manifest, bs, use_cuda)
    char2idx, idx2char = _attach_vocab(ds_tr, ds_va)
    save_vocab(os.path.join(out_dir, "vocab.json"), idx2char)
    print(f"[train_vsr] vocab size={len(idx2char)} (blank index assumed=0)")

    model = CNN3D_BiGRU(vocab_size=len(idx2char)).to(device)
    opt = AdamW(model.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = GradScaler(enabled=use_cuda)
    best = float("inf")
    skipped_tr_total, skipped_va_total = 0, 0

    for ep in range(1, epochs + 1):
        # ---------- Train ----------
        model.train()
        tr_loss, tr_steps, skipped_tr = 0.0, 0, 0
        for xb, xlen, yb, ylen in dl_tr:
            xb, yb, ylen, xlen = xb.to(device, non_blocking=use_cuda), yb.to(device), ylen.to(device), xlen.to(device)
            opt.zero_grad(set_to_none=True)
            if use_cuda:
                with autocast():
                    logits = model(xb) # [B, T, V]
                    T = logits.shape[1]
                    mask = (ylen <= T)
                    yb_sub, ylen_sub = _subset_flat_targets(yb, ylen, mask)
                    skipped_tr += int((~mask).sum().item())
                    if ylen_sub.numel() == 0:
                        continue
                    in_lens = torch.full((ylen_sub.shape[0],), T, dtype=torch.long, device=device)
                    logp = logits[mask].log_softmax(dim=-1) # [B_sub, T, V]
                    loss = ctc(logp.transpose(0, 1), yb_sub, in_lens, ylen_sub)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                T = logits.shape[1]
                mask = (ylen <= T)
                yb_sub, ylen_sub = _subset_flat_targets(yb, ylen, mask)
                skipped_tr += int((~mask).sum().item())
                if ylen_sub.numel() == 0:
                    continue
                in_lens = torch.full((ylen_sub.shape[0],), T, dtype=torch.long, device=device)
                logp = logits[mask].log_softmax(dim=-1)
                loss = ctc(logp.transpose(0, 1), yb_sub, in_lens, ylen_sub)
                loss.backward()
                opt.step()
            tr_loss += float(loss.item())
            tr_steps += 1
        tr_loss = tr_loss / max(1, tr_steps)
        skipped_tr_total += skipped_tr

        # ---------- Val ----------
        model.eval()
        va_loss, va_steps, skipped_va = 0.0, 0, 0
        with torch.no_grad():
            for xb, xlen, yb, ylen in dl_va:
                xb, yb, ylen, xlen = xb.to(device, non_blocking=use_cuda), yb.to(device), ylen.to(device), xlen.to(device)
                logits = model(xb)
                T = logits.shape[1]
                mask = (ylen <= T)
                yb_sub, ylen_sub = _subset_flat_targets(yb, ylen, mask)
                skipped_va += int((~mask).sum().item())
                if ylen_sub.numel() == 0:
                    continue
                in_lens = torch.full((ylen_sub.shape[0],), T, dtype=torch.long, device=device)
                logp = logits[mask].log_softmax(dim=-1)
                loss = ctc(logp.transpose(0, 1), yb_sub, in_lens, ylen_sub)
                va_loss += float(loss.item())
                va_steps += 1
        va_loss = va_loss / max(1, va_steps)
        skipped_va_total += skipped_va

        print(f"Epoch {ep}: train {tr_loss:.3f} | val {va_loss:.3f} | skipped(train/val) {skipped_tr}/{skipped_va}")

        if va_loss < best and va_steps > 0:
            best = va_loss
            torch.save({"state_dict": model.state_dict(), "idx2char": idx2char},
                       os.path.join(out_dir, "lip_reader.pt"))
            print(f"Saved best to {os.path.join(out_dir, 'lip_reader.pt')}")

    print(f"[train_vsr] total skipped → train {skipped_tr_total}, val {skipped_va_total}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_manifest")
    ap.add_argument("val_manifest")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = dev.startswith("cuda")
    print(f"[train_vsr] device={dev} | train={args.train_manifest} | val={args.val_manifest}")
    train(args.train_manifest, args.val_manifest, out_dir=args.out_dir, epochs=args.epochs, bs=args.bs, lr=args.lr, device=dev)
