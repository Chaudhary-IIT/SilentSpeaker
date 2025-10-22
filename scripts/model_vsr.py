# scripts/model_vsr.py
import torch
import torch.nn as nn


class CNN3D_BiGRU(nn.Module):
    """
    Preserves the time axis (T). Stride/pool only on H,W.
    3D CNN -> global average over H,W -> BiGRU over T -> classifier.
    """

    def __init__(self, vocab_size: int, in_ch: int = 1, rnn_hidden: int = 256):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(0.1)
        self.rnn = nn.GRU(input_size=128, hidden_size=rnn_hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(rnn_hidden * 2, vocab_size)

    def _to_c_t_h_w(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B,T,1,H,W] or [B,T,H,W]; return [B,1,T,H,W]
        if x.dim() == 5:
            # [B, T, C, H, W] -> [B, C, T, H, W]
            return x.permute(0, 2, 1, 3, 4).contiguous()
        elif x.dim() == 4:
            x = x.unsqueeze(2)  # [B,T,1,H,W]
            return x.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_c_t_h_w(x)       # [B,1,T,H,W]

        x = self.conv1(x)             # [B,32,T,H1,W1]
        x = self.conv2(x)             # [B,64,T,H2,W2]
        x = self.conv3(x)             # [B,128,T,H3,W3]
        x = self.dropout(x)

        # Global average over spatial dims only -> [B,128,T]
        x = x.mean(dim=[3, 4])
        # RNN expects [B,T,feat]
        x = x.transpose(1, 2).contiguous()  # [B,T,128]

        x, _ = self.rnn(x)            # [B,T,2*hidden]
        logits = self.classifier(x)    # [B,T,V]
        return logits
