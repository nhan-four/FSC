import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "iot_data.csv"  # ĐỔI THÀNH TÊN FILE CSV CỦA BẠN
TEST_SIZE = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 80
LR = 1e-3
SNR_DB = 10.0          # SNR cho kênh fading
LATENT_DIM = 16        # kích thước semantic latent
CHAN_DIM = 16          # kích thước vector sau ChannelEncoder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. LOAD & PREPROCESS CSV (THEO FORMAT BẠN GỬI)
# ============================================================

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Các cột số mình dùng làm "sensor / feature" cho semantic encoder
    numeric_cols = [
        "NDVI", "NDRE", "RGB_Damage_Score",
        "N", "P", "K",
        "Moisture", "pH", "Temperature", "Humidity",
        "Energy_Consumed_mAh", "Latency_ms"
    ]

    # Ép float, xử lý NaN
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Bỏ các dòng thiếu quá nặng, ở đây yêu cầu N,P,K phải có
    df = df.dropna(subset=["N", "P", "K"])

    # Fill NaN còn lại bằng mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    X = df[numeric_cols].values.astype("float32")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")

    X_train, X_test = train_test_split(
        X_scaled, test_size=TEST_SIZE, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, X_train_t)  # autoencoder: target = input
    test_ds = TensorDataset(X_test_t, X_test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return numeric_cols, scaler, train_loader, test_loader, X_train.shape[1]


# ============================================================
# 2. MÔ HÌNH NHƯ PAPER (BẢN CHO VECTOR CSV)
# ============================================================

class SemanticEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        # x: [B, input_dim]
        return self.net(x)


class SemanticDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z_hat):
        # z_hat: [B, latent_dim]
        return self.net(z_hat)


class ChannelEncoder(nn.Module):
    def __init__(self, latent_dim, chan_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, chan_dim),
            nn.Tanh()    # giới hạn năng lượng (giống JSCC)
        )

    def forward(self, z):
        # z: [B, latent_dim]
        return self.net(z)


class ChannelDecoder(nn.Module):
    def __init__(self, chan_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chan_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, y_eq):
        # y_eq: [B, chan_dim]
        return self.net(y_eq)


# --------------------------
# ADNet (CSI denoiser, đơn giản hóa)
# --------------------------
class ADNetBlock(nn.Module):
    """Một block conv1d đơn giản để dùng trong ADNet"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, L]
        out = self.conv(x)
        out = out + x  # residual
        return self.relu(out)


class ADNet(nn.Module):
    """
    Attention-guided Denoising Network (phiên bản nhẹ cho vector H_ls)
    Input:  H_ls dạng [B, L] (L = số chiều chan_dim)
    Output: H_refine cùng shape
    """
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.in_conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.block1 = ADNetBlock(16)
        self.block2 = ADNetBlock(16)
        self.block3 = ADNetBlock(16)

        # "attention" đơn giản: global pooling + FC
        self.attention_fc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv1d(16, 1, kernel_size=3, padding=1)

    def forward(self, h_ls):
        # h_ls: [B, L]  -> [B, 1, L]
        x = h_ls.unsqueeze(1)
        x = self.in_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # attention theo channel dimension
        # global avg pooling trên chiều L
        gap = x.mean(dim=2)       # [B, 16]
        att = self.attention_fc(gap)  # [B, 16]
        att = att.unsqueeze(-1)   # [B, 16, 1]
        x = x * att              # scale mỗi channel

        x = self.out_conv(x)     # [B, 1, L]
        out = x.squeeze(1)       # [B, L]
        return out


# --------------------------
# Kênh fading + CSI + ADNet
# --------------------------
class FadingChannelWithCSI(nn.Module):
    """
    Kênh Rayleigh fading cho mỗi chiều của x_chan:
        y = H_true * x + n

    CSI ước lượng: H_ls = H_true + noise_est
    Sau đó refine bằng ADNet -> H_refine
    Equalization:
        y_eq = y / H_refine
    """
    def __init__(self, chan_dim, snr_db=10.0):
        super().__init__()
        self.chan_dim = chan_dim
        self.snr_db = snr_db
        self.adnet = ADNet(length=chan_dim)

    def forward(self, x_chan):
        """
        x_chan: [B, D]  (D = chan_dim)
        """
        B, D = x_chan.shape
        device = x_chan.device

        # Rayleigh fading H_true: |N(0,1)|
        H_true = torch.abs(torch.randn(B, D, device=device)) + 1e-3  # tránh 0

        # Noise AWGN (channel noise)
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_var = 1.0 / snr_linear
        noise_std = noise_var ** 0.5
        noise = noise_std * torch.randn_like(x_chan)

        # Forward qua kênh
        y = H_true * x_chan + noise

        # Ước lượng CSI (LS fake): H_ls = H_true + noise_est
        est_noise_std = 0.2   # độ nhiễu của ước lượng CSI (bạn có thể chỉnh)
        H_ls = H_true + est_noise_std * torch.randn_like(H_true)

        # Refine CSI bằng ADNet
        H_refine = self.adnet(H_ls)

        # Equalization
        y_eq = y / (H_refine + 1e-3)

        return y_eq, H_true, H_ls, H_refine


# --------------------------
# Toàn bộ L-DeepSC cho CSV
# --------------------------
class LDeepSC_CSV(nn.Module):
    def __init__(self, input_dim, latent_dim, chan_dim, snr_db):
        super().__init__()
        self.semantic_encoder = SemanticEncoder(input_dim, latent_dim)
        self.semantic_decoder = SemanticDecoder(latent_dim, input_dim)
        self.channel_encoder = ChannelEncoder(latent_dim, chan_dim)
        self.channel_decoder = ChannelDecoder(chan_dim, latent_dim)
        self.channel = FadingChannelWithCSI(chan_dim, snr_db)

    def forward(self, x):
        """
        x: [B, input_dim]
        """
        # Semantic encoder
        z = self.semantic_encoder(x)

        # Channel encoder
        x_chan = self.channel_encoder(z)

        # Fading channel + CSI + ADNet
        y_eq, H_true, H_ls, H_refine = self.channel(x_chan)

        # Channel decoder
        z_hat = self.channel_decoder(y_eq)

        # Semantic decoder
        x_hat = self.semantic_decoder(z_hat)

        return x_hat, {
            "H_true": H_true,
            "H_ls": H_ls,
            "H_refine": H_refine,
            "x_chan": x_chan,
            "y_eq": y_eq,
        }


# ============================================================
# 3. TRAINING LOOP
# ============================================================

def train_model(train_loader, test_loader, input_dim):
    model = LDeepSC_CSV(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        chan_dim=CHAN_DIM,
        snr_db=SNR_DB
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            x_hat, extra = model(xb)
            loss = criterion(x_hat, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- Eval ----
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                x_hat, _ = model(xb)
                loss = criterion(x_hat, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Test MSE: {test_loss:.4f}"
        )

    return model


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {CSV_PATH}")

    numeric_cols, scaler, train_loader, test_loader, input_dim = load_and_preprocess(CSV_PATH)
    print("Numeric cols dùng cho model:", numeric_cols)
    print("Input dim:", input_dim)

    model = train_model(train_loader, test_loader, input_dim)

    # Demo: truyền 1 mẫu qua semantic + kênh + decoder
    print("\nDemo 1 sample qua hệ thống L-DeepSC_CSV:")
    test_batch = next(iter(test_loader))[0][:1].to(DEVICE)  # 1 sample
    with torch.no_grad():
        x_hat, extras = model(test_batch)

    x_original = scaler.inverse_transform(test_batch.cpu().numpy())
    x_recon = scaler.inverse_transform(x_hat.cpu().numpy())

    print("Original (unscaled):", x_original[0])
    print("Reconstructed      :", x_recon[0])

    # Bạn có thể lưu model nếu muốn
    torch.save(model.state_dict(), "l_deepsc_csv.pth")
    print("\nModel đã được lưu vào l_deepsc_csv.pth")


if __name__ == "__main__":
    main()
