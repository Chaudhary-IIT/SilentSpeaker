# scripts/model_vsr.py
import torch, torch.nn as nn

class CNN3D_BiGRU(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 32, (3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, (3,3,3), stride=(1,2,2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, (3,3,3), stride=(1,2,2), padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # keep time, pool H,W -> 1
        self.rnn = nn.GRU(128, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(512, vocab_size)

    def forward(self, x):
        # x: (B,T,1,H,W)
        x = x.transpose(1,2)          # (B,1,T,H,W) -> (B,1,T,H,W) for Conv3d expects (B,C,T,H,W)
        feat = self.backbone(x)        # (B,128,T',H',W')
        feat = self.pool(feat)         # (B,128,T',1,1)
        feat = feat.squeeze(-1).squeeze(-1).transpose(1,2)   # (B,T',128)
        out, _ = self.rnn(feat)        # (B,T',512)
        logits = self.classifier(out)  # (B,T',V)
        return logits
