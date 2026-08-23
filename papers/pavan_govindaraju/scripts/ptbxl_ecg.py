"""Fit a Neural SDE to a dense continuous heart rate trajectory from a PTB-XL ECG recording."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from nodefit.constants import DEVICE
from nodefit.neural_sde import NeuralSDE, SDE
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from torchsde import sdeint

PAPER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PAPER_DIR / "data"
RESULTS_DIR = PAPER_DIR / "results"
RECORD = DATA_DIR / "16834_hr"
LEAD = "II"
TRAIN_END_S = 6.0
RECORD_END_S = 10.0
N_EPOCHS = 800
BATCH_SIZE = 35
DT = 0.05

torch.manual_seed(42)
np.random.seed(42)


def read_wfdb_lead(record_path, lead_name):
    """Read a specific lead from a WFDB format pair (.hea / .dat)."""
    header_path = Path(str(record_path) + ".hea")
    data_path = Path(str(record_path) + ".dat")
    lines = header_path.read_text().splitlines()
    _name, n_signals, fs, n_samples = lines[0].split()[:4]
    n_signals, fs, n_samples = int(n_signals), float(fs), int(n_samples)

    leads, gains, baselines = [], [], []
    for line in lines[1 : 1 + n_signals]:
        parts = line.split()
        gain_token = parts[2]
        gain = float(gain_token.split("(")[0].split("/")[0])
        baseline = 0.0
        if "(" in gain_token:
            baseline = float(gain_token.split("(")[1].split(")")[0])
        leads.append(parts[-1])
        gains.append(gain)
        baselines.append(baseline)

    raw = np.fromfile(data_path, dtype="<i2").reshape(n_samples, n_signals)
    index = leads.index(lead_name)
    physical = (raw[:, index].astype(np.float64) - baselines[index]) / gains[index]
    time_arr = np.arange(n_samples, dtype=np.float64) / fs
    return time_arr, physical, fs


def cubic(t, a, b, c, d):
    return a + b * t + c * t**2 + d * t**3


class FastSDE(SDE):
    def f(self, t, y):
        t_tensor = y.new_full((y.shape[0], 1), t)
        return self.drift_nn(torch.cat((t_tensor, y), dim=1))

    def g(self, t, y):
        t_tensor = y.new_full((y.shape[0], 1), t)
        return self.diffusion_nn(torch.cat((t_tensor, y), dim=1))


class FastNeuralSDE(NeuralSDE):
    def __init__(self, drift_nn, diffusion_nn, t, data, batch_size=35, dt=0.05):
        super().__init__(drift_nn, diffusion_nn, t, data, batch_size)
        self.sde = FastSDE(drift_nn, diffusion_nn)
        self.dt = dt
        self.t = self.t.to(dtype=torch.float32, device=DEVICE)
        self.data = self.data.to(dtype=torch.float32, device=DEVICE)
        self.y0 = self.y0.to(dtype=torch.float32, device=DEVICE)
        self.drift_nn = drift_nn.to(dtype=torch.float32, device=DEVICE)
        self.diffusion_nn = diffusion_nn.to(dtype=torch.float32, device=DEVICE)

    def loss(self):
        self.nn_data = sdeint(
            self.sde, self.y0, self.t, method=self.sde.numerical_method, dt=self.dt
        )
        mu = self.nn_data.mean(dim=1)
        diff = self.data - mu
        return torch.mean(diff**2)

    def train(self, num_epochs=600):
        for i in range(num_epochs):
            self.sde.drift_opt.zero_grad()
            self.sde.diffusion_opt.zero_grad()
            current_loss = self.loss()
            current_loss.backward()
            self.sde.drift_opt.step()
            self.sde.diffusion_opt.step()

    def extrapolate_dense(self, t_dense, n_samples=150):
        y0_eval = self.y0[0:1].repeat(n_samples, 1)
        return sdeint(
            self.sde,
            y0_eval,
            torch.tensor(t_dense, dtype=torch.float32, device=DEVICE),
            method=self.sde.numerical_method,
            dt=self.dt,
        )


def run_ptbxl_example():
    t_raw, lead_ii, fs = read_wfdb_lead(RECORD, LEAD)

    # Detect R-peaks and extract instantaneous Heart Rate (BPM)
    peaks, _ = find_peaks(lead_ii, prominence=0.3, distance=int(0.35 * fs))
    peak_times = t_raw[peaks]
    rr_intervals = np.diff(peak_times)
    rr_midpoints = (peak_times[:-1] + peak_times[1:]) / 2.0
    hr_inst = 60.0 / rr_intervals

    # Continuous instantaneous heart rate trajectory via PCHIP interpolation
    pchip = PchipInterpolator(rr_midpoints, hr_inst)
    t_dense_all = np.linspace(rr_midpoints[0], rr_midpoints[-1], 90)
    hr_dense_all = pchip(t_dense_all)

    # Training window: t <= 6.0s (N=56 points)
    train_mask = t_dense_all <= TRAIN_END_S
    t_train = t_dense_all[train_mask]
    y_train = (hr_dense_all[train_mask] / 100.0).reshape(-1, 1)

    # Forecast / test window: t > 6.0s (N=34 points)
    test_mask = t_dense_all > TRAIN_END_S
    t_test = t_dense_all[test_mask]
    y_test = (hr_dense_all[test_mask] / 100.0).reshape(-1, 1)

    print(f"Dense continuous HR curve: {len(t_train)} train points, {len(t_test)} test points")

    # Multi-harmonic periodic coordinates for bounded autonomic oscillatory dynamics
    W0 = 2.0 * np.pi * 0.35  # fundamental frequency ~ 0.35 Hz (respiratory period ~ 2.85s)
    K_HARMONICS = 3

    class PeriodicStationarySDE(SDE):
        def __init__(self, drift_nn, diffusion_nn, k_harmonics=3, w0=W0):
            super().__init__(drift_nn, diffusion_nn)
            self.k_harmonics = k_harmonics
            self.w0 = w0

        def _features(self, t, y):
            # Bounded periodic harmonic phase coordinates + state y (prevents unconstrained linear runaway)
            t_col = y.new_full((y.shape[0], 1), t)
            feats = []
            for k in range(1, self.k_harmonics + 1):
                feats.append(torch.sin((k * self.w0) * t_col))
                feats.append(torch.cos((k * self.w0) * t_col))
            feats.append(y)
            return torch.cat(feats, dim=1)

        def f(self, t, y):
            return self.drift_nn(self._features(t, y))

        def g(self, t, y):
            # Diffusion network outputting realistic physiological AF volatility (14-25 BPM)
            raw = self.diffusion_nn(self._features(t, y))
            return 0.18 + 0.12 * torch.sigmoid(raw)

    # Feature dimension = 2*K_HARMONICS + 1 = 7
    in_dim = 2 * K_HARMONICS + 1
    drift_nn = nn.Sequential(
        nn.Linear(in_dim, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )
    diffusion_nn = nn.Sequential(
        nn.Linear(in_dim, 24),
        nn.Tanh(),
        nn.Linear(24, 1),
    )

    started = time.perf_counter()
    sde = FastNeuralSDE(
        drift_nn, diffusion_nn, t_train, y_train, batch_size=BATCH_SIZE, dt=DT
    )
    sde.sde = PeriodicStationarySDE(drift_nn, diffusion_nn, k_harmonics=K_HARMONICS, w0=W0)
    sde.sde.drift_opt = torch.optim.Adam(drift_nn.parameters(), lr=0.008)
    sde.sde.diffusion_opt = torch.optim.Adam(diffusion_nn.parameters(), lr=0.008)
    sde.train(num_epochs=600)
    train_time = time.perf_counter() - started

    t_eval = np.linspace(t_train[0], t_dense_all[-1], 250)
    full_sim = sde.extrapolate_dense(t_eval, n_samples=150).detach().cpu().numpy()
    elapsed = time.perf_counter() - started
    print(f"Wall-clock time (train + extrapolate): {elapsed:.2f} s")

    mean_dense = full_sim[:, :, 0].mean(axis=1) * 100.0
    std_dense = full_sim[:, :, 0].std(axis=1) * 100.0

    # Baseline: cubic curve_fit
    popt, _ = curve_fit(cubic, t_train, y_train[:, 0] * 100.0, p0=[100.0, 0.0, 0.0, 0.0])
    base_eval = cubic(t_eval, *popt)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # (a) Raw clinical ECG
    fig1, ax1 = plt.subplots(figsize=(8.5, 2.5))
    ax1.plot(t_raw, lead_ii, color="#2d3748", lw=0.85, label="PTB-XL Record #16834 (Lead II, 500 Hz)")
    ax1.plot(peak_times, lead_ii[peaks], "r^", markersize=4.5, label="Detected R-peaks")
    ax1.axvline(TRAIN_END_S, color="0.45", linestyle="-.", lw=1.2, label=r"Forecast horizon ($t=6$s)")
    ax1.set_ylabel("Voltage (mV)", fontsize=9)
    ax1.set_xlabel("Time (seconds)", fontsize=9)
    ax1.set_xlim(0, RECORD_END_S)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    out1 = RESULTS_DIR / "ptbxl_ecg_raw.png"
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # (b) Continuous Heart Rate SDE
    fig2, ax2 = plt.subplots(figsize=(8.5, 3.8))
    ax2.plot(t_train, y_train[:, 0] * 100.0, "k.", markersize=4.5, label=r"Training Trajectory ($t \leq 6$s, $N=56$)")
    ax2.plot(t_test, y_test[:, 0] * 100.0, "s", color="#2b6cb0", markersize=3.5, label=r"Held-out Future Trajectory ($t > 6$s, $N=34$)")
    ax2.plot(rr_midpoints, hr_inst, "o", color="#e53e3e", markersize=5.0, fillstyle="none", markeredgewidth=1.2, label="Discrete Beat Markers")
    ax2.plot(t_eval, base_eval, "r--", lw=1.8, label="curve_fit baseline (cubic polynomial)")
    ax2.plot(t_eval, mean_dense, color="#2b6cb0", lw=2.2, label="Neural SDE learned drift (mean)")
    ax2.fill_between(
        t_eval,
        mean_dense - std_dense,
        mean_dense + std_dense,
        color="#63b3ed",
        alpha=0.35,
        label=r"Neural SDE diffusion ($\pm 1\sigma$ band)",
    )
    ax2.fill_between(
        t_eval,
        mean_dense - 2 * std_dense,
        mean_dense + 2 * std_dense,
        color="#bee3f8",
        alpha=0.25,
        label=r"Neural SDE diffusion ($\pm 2\sigma$ band)",
    )
    ax2.axvline(TRAIN_END_S, color="0.45", linestyle="-.", lw=1.2)
    ax2.set_xlabel("Time (seconds)", fontsize=9.5)
    ax2.set_ylabel("Instantaneous Heart Rate (BPM)", fontsize=9.5)
    ax2.set_xlim(0, RECORD_END_S)
    ax2.set_ylim(50, 180)
    ax2.legend(loc="upper left", fontsize=7.8, ncol=2)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = RESULTS_DIR / "ptbxl_hrv_sde.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Generated clean figures:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    run_ptbxl_example()
