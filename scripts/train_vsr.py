# scripts/train_vsr.py
import os, torch, torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from dataset_vsr import MouthROIVideoDataset, pad_collate
from model_vsr import CNN3D_BiGRU
from vocab import build_vocab, save_vocab

# scripts/train_vsr.py (patch the top-level train() signature and device selection)

def train(manifest_train, manifest_val, out_dir="checkpoints", epochs=5, bs=4, lr=1e-3, device=None):
    import torch
    # Choose CPU unless CUDA is actually available
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    os.makedirs(out_dir, exist_ok=True)
    char2idx, idx2char = build_vocab()
    save_vocab(os.path.join(out_dir, "vocab.json"), idx2char)

    ds_tr = MouthROIVideoDataset(manifest_train, char2idx=char2idx)
    ds_va = MouthROIVideoDataset(manifest_val, char2idx=char2idx)

    # On Windows, num_workers=0 is safer
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0, collate_fn=pad_collate)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=0, collate_fn=pad_collate)

    model = CNN3D_BiGRU(vocab_size=len(idx2char)).to(device)
    opt = AdamW(model.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    # Disable AMP on CPU
    use_amp = device.startswith("cuda")
    scaler = GradScaler(enabled=use_amp)

    best = 1e9
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, xlen, yb, ylen in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with autocast(enabled=True):
                    logits = model(xb)
                    logp = logits.log_softmax(dim=-1)
                    T = logp.shape[1]
                    input_lengths = xlen.clamp(max=T)
                    loss = ctc(logp.transpose(0,1), yb, input_lengths, ylen)
                scaler.scale(loss).step(opt)
                scaler.update()
            else:
                logits = model(xb)
                logp = logits.log_softmax(dim=-1)
                T = logp.shape[1]
                input_lengths = xlen.clamp(max=T)
                loss = ctc(logp.transpose(0,1), yb, input_lengths, ylen)
                loss.backward()
                opt.step()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(dl_tr))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, xlen, yb, ylen in dl_va:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                logp = logits.log_softmax(dim=-1)
                T = logp.shape[1]
                input_lengths = xlen.clamp(max=T)
                loss = ctc(logp.transpose(0,1), yb, input_lengths, ylen)
                va_loss += float(loss.item())
        va_loss /= max(1, len(dl_va))
        print(f"Epoch {ep}: train {tr_loss:.3f} | val {va_loss:.3f}")

        if va_loss < best:
            best = va_loss
            torch.save({"state_dict": model.state_dict(), "idx2char": idx2char},
                       os.path.join(out_dir, "lip_reader.pt"))
            print(f"Saved best to {os.path.join(out_dir,'lip_reader.pt')}")

if __name__ == "__main__":
    import sys, torch
    # Force CPU if CUDA is not available
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train(sys.argv[1], sys.argv[2], device=dev)
