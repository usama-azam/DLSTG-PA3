# %% [markdown]
# # AI651 — Deep Learning for Space, Time and Graphs — Programming Assignment 3
#
# **Spatio-temporal traffic forecasting on LargeST-SD with fixed, adaptive and hybrid graphs.**
#
# This single notebook reproduces every numerical result and figure in the report:
#
# | Part | What it does |
# |---|---|
# | Q1 | NumPy verification of the aggregation / injectivity arguments (affine message, base-(K+1) encoding, sum/mean/max table) |
# | Q2 | Ring-road experiments: least-squares fit of a symmetric layer, 20-step roll-out heatmaps, directional kernel, history-based layer |
# | Q3 | LargeST San Diego extraction (716 sensors, 2020–2021), 5 → 15 min resampling, missing values and zero-flow faults, road-graph analysis, 1/2-hop neighbourhoods |
# | Q4 | Forecasting windows (8 → 4), chronological splits, temporal-conv + GCN + head model, temporal-only baseline |
# | Q5 | Fixed vs learnable (adaptive) vs hybrid adjacency, identical protocol, **3 seeds each**, plus 4 ablations |
# | Q6 | Comparison tables (mean ± std over seeds), predicted-vs-actual plots, learned-graph analysis across seeds |
# | Q7 | Hyper-parameters, seeds, hardware, memory and timings are printed and saved to `outputs/` |
#
# The 24 training runs (8 configurations × 3 seeds) are executed by `stgnn_lib.py` worker processes, **one per GPU**
# (Kaggle provides 2 × T4), so every run has a dedicated GPU and its training time is not distorted by sharing.
# If the dataset is not found, the notebook falls back to a small **synthetic smoke test** of the whole pipeline.

# %%
import os, sys, json, time, math, random, platform, warnings, subprocess, shutil
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

# ----------------------------------------------------------------------------- paths
CANDIDATE_DIRS = ["/kaggle/input/datasets/liuxu77/largest", "/kaggle/input/largest",
                  os.environ.get("LARGEST_DIR", "data/largest")]
DATA_DIR = next((d for d in CANDIDATE_DIRS if os.path.exists(os.path.join(d, "ca_meta.csv"))), None)
SMOKE = DATA_DIR is None
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
OUT_DIR = os.path.join(WORK, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RUN_DIR = os.path.join(OUT_DIR, "runs")
TMP_DIR = "/tmp/dlstg_pa3" if os.path.exists("/tmp") else os.path.join(WORK, "tmp")
for d in (FIG_DIR, RUN_DIR, TMP_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------------- config
CFG = dict(
    seeds=[42, 43, 44],          # every configuration is trained with each seed
    district=11,                 # San Diego = Caltrans District 11
    years=[2020, 2021],
    agg="sum",                   # PeMS 5-min flow is a vehicle COUNT per interval -> sum to 15 min
    t_in=8, t_out=4,             # 2 h history -> 1 h horizon at 15-min resolution
    split=dict(train=("2020-01-01", "2020-12-31 23:45"),
               val=("2021-01-01", "2021-03-31 23:45"),
               test=("2021-04-01", "2021-12-31 23:45")),
    zero_run_steps=4,            # >= 4 consecutive zero 15-min totals (>= 1 h) = detector fault -> missing
    short_gap_steps=4,           # input gaps <= 1 h: forward-fill (causal); longer: 2020 time-of-week profile
    src_min_obs=0.5,             # sensors with < 50% observed 2020 data are not used as message sources
    hidden=32,                   # d (assignment: 16-32)
    gcn_layers=2,
    emb_dim=10,                  # d_e of the node embeddings E1, E2
    topk=20,                     # sparsify adaptive adjacency to top-k neighbours per node
    alpha_init=0.5,              # hybrid: alpha = sigmoid(a), learnable, initialised at 0.5
    batch_size=64,
    lr=1e-3, lr_factor=0.5, lr_patience=3, min_lr=1e-5,   # Adam + ReduceLROnPlateau on val MAE
    weight_decay=0.0,
    max_epochs=80,
    patience=10,                 # early stopping on validation MAE
    grad_clip=5.0,
    loss="masked MAE",
)
if SMOKE:
    CFG.update(max_epochs=1, patience=1)   # keep all 3 seeds so the multi-seed analysis is exercised
SEED = CFG["seeds"][0]
print("DATA_DIR:", DATA_DIR, "| SMOKE TEST:", SMOKE, "| OUT_DIR:", OUT_DIR)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path)
    print("saved", path)


def peak_rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # KB -> GB on Linux
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e9            # Windows
        except Exception:
            return float("nan")

# %% [markdown]
# ## Hardware and software (Q7)

# %%
def hardware_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "cpu": platform.processor() or "unknown", "cpu_count": os.cpu_count()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_count=torch.cuda.device_count(),
                    gpu_mem_gb=round(p.total_memory / 1e9, 1), cuda=torch.version.cuda)
    return info


HW = hardware_info()
for k, v in HW.items():
    print(f"{k:>10}: {v}")

# %% [markdown]
# ---
# # Question 1 — What does a sum remember?
#
# ### (a) Affine message
# With $\phi(x)=wx+c$: $\sum_{u\in N(v)}\phi(x_u) = w S_v + c\, n_v$, so $h_v=\rho(wS_v+cn_v)$ depends on the
# neighbourhood only through $(n_v, S_v)$. Two neighbourhoods with equal $(n_v,S_v)$ but a different number
# of steady neighbours: $\{-1,+1,0\}$ (one steady) and $\{0,0,0\}$ (three steady), both $n_v=3, S_v=0$.

# %%
N1, N2 = np.array([-1, +1, 0]), np.array([0, 0, 0])
rng = np.random.default_rng(0)
for _ in range(3):
    w, c = rng.normal(size=2)
    rho = lambda z: np.tanh(3 * z) + z ** 2          # an arbitrary nonlinearity
    a1, a2 = (w * N1 + c).sum(), (w * N2 + c).sum()
    print(f"w={w:+.3f} c={c:+.3f}  sum1={a1:+.4f} sum2={a2:+.4f}  rho1={rho(a1):+.4f} rho2={rho(a2):+.4f}")
print("steady counts:", (N1 == 0).sum(), "vs", (N2 == 0).sum())

# %% [markdown]
# ### (b) Base-(K+1) encoding
# Choose $\phi(-1)=1,\ \phi(0)=K+1,\ \phi(+1)=(K+1)^2$. Then
# $s=\sum\phi = n_- + n_0(K+1) + n_+(K+1)^2$ with every count $\le K < K+1$, i.e. the three counts are the
# digits of $s$ in base $K+1$ (no carries). Decoding: $n_-=s \bmod (K+1)$, $n_0=\lfloor s/(K+1)\rfloor \bmod (K+1)$,
# $n_+=\lfloor s/(K+1)^2\rfloor$. Non-steady count: $\phi(x)=x^2$ (or $|x|$) gives $\sum\phi=n_-+n_+$.

# %%
K = 8
B_ = K + 1
phi = {-1: 1, 0: B_, +1: B_ ** 2}
# every triple of counts (n_minus, n_zero, n_plus) with 1 <= total <= K
triples = np.array([(a, b, c) for a in range(K + 1) for b in range(K + 1) for c in range(K + 1)
                    if 1 <= a + b + c <= K])
# build the actual neighbourhood (multiset of trends) and sum its messages
sums = np.array([sum(phi[x] for x in [-1] * a + [0] * b + [1] * c) for a, b, c in triples])
dec = np.stack([sums % B_, (sums // B_) % B_, sums // B_ ** 2], axis=1)
print("number of count triples           :", len(triples))
print("all sums distinct                 :", len(np.unique(sums)) == len(sums))
print("decoding recovers every triple    :", np.array_equal(dec, triples))
nonsteady = np.array([sum(x ** 2 for x in [-1] * a + [0] * b + [1] * c) for a, b, c in triples])
print("phi(x)=x^2 counts non-steady      :", np.array_equal(nonsteady, triples[:, 0] + triples[:, 2]))
print("example: counts (2,1,3) -> sum", 2 * 1 + 1 * B_ + 3 * B_ ** 2)

# %% [markdown]
# ### (c) Sum vs mean vs max with one-hot messages — brute-force check
# For every task/method we enumerate **all** neighbourhoods of size 1–8 and test whether the aggregated vector
# determines the answer (i.e. no two neighbourhoods share a vector but differ in the answer). A collision is
# printed as the counter-example.

# %%
from itertools import combinations_with_replacement

onehot = {-1: np.array([1, 0, 0]), 0: np.array([0, 1, 0]), 1: np.array([0, 0, 1])}
aggs = {"Sum": lambda M: M.sum(0), "Mean": lambda M: M.mean(0), "Max": lambda M: M.max(0)}
tasks = {"(i) count steady": lambda nb: int(sum(1 for x in nb if x == 0)),
         "(ii) >half slowing": lambda nb: sum(1 for x in nb if x == -1) > len(nb) / 2,
         "(iii) any slowing": lambda nb: any(x == -1 for x in nb)}
neighbourhoods = [nb for n in range(1, 9) for nb in combinations_with_replacement([-1, 0, 1], n)]
print("neighbourhoods enumerated:", len(neighbourhoods))
q1c_rows = []
for tname, task in tasks.items():
    for aname, agg in aggs.items():
        seen, counter = {}, None
        for nb in neighbourhoods:
            key = tuple(round(float(v), 9) for v in agg(np.stack([onehot[x] for x in nb])))
            ans = task(nb)
            if key in seen and seen[key][1] != ans:
                counter = (key, seen[key], (nb, ans)); break
            seen.setdefault(key, (nb, ans))
        ok = counter is None
        q1c_rows.append((tname, aname, "Yes" if ok else "No",
                         "" if ok else f"vec={counter[0]}: {counter[1][0]}->{counter[1][1]} vs {counter[2][0]}->{counter[2][1]}"))
print(pd.DataFrame(q1c_rows, columns=["task", "method", "possible", "first collision found"]).to_string(index=False))

# %% [markdown]
# ---
# # Question 2 — Which way is the slowdown moving?
# ### (a)(i) Least-squares fit of the symmetric layer
# With $x_{20}=1$: $\hat x_{19}=b,\ \hat x_{20}=a,\ \hat x_{21}=b$ and targets $1,0,0$.
# Loss $=(b-1)^2+a^2+b^2 \Rightarrow a=0,\ b=\tfrac12$, minimum loss $\tfrac12$.

# %%
Nring = 40
x0 = np.zeros(Nring); x0[20] = 1.0
x1 = np.roll(x0, -1)                                     # true next snapshot
D = np.stack([x0, np.roll(x0, 1) + np.roll(x0, -1)], 1)  # columns: x_i, x_{i-1}+x_{i+1}
(a_ls, b_ls), *_ = np.linalg.lstsq(D, x1, rcond=None)
print(f"least squares: a = {a_ls:.4f}, b = {b_ls:.4f}, SSE = {((D @ [a_ls, b_ls] - x1) ** 2).sum():.4f}")

# %% [markdown]
# ### (a)(ii) 20-step roll-out of the fitted symmetric layer vs. the true motion

# %%
idx = np.arange(Nring)
bump = np.exp(-(idx - 20) ** 2 / 8)
pred_seq, true_seq = [bump.copy()], [bump.copy()]
for _ in range(20):
    p = pred_seq[-1]
    pred_seq.append(a_ls * p + b_ls * (np.roll(p, 1) + np.roll(p, -1)))
    true_seq.append(np.roll(true_seq[-1], -1))
pred_seq, true_seq = np.array(pred_seq), np.array(true_seq)
vmin, vmax = 0, 1
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a_, s_, t_ in [(ax[0], true_seq, "True motion  $x_i(t+1)=x_{i+1}(t)$"),
                   (ax[1], pred_seq, f"Symmetric layer  a={a_ls:.2f}, b={b_ls:.2f}")]:
    im = a_.imshow(s_, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
                   extent=[-0.5, Nring - 0.5, -0.5, 20.5])
    a_.set_xlabel("sensor index i"); a_.set_title(t_); a_.grid(False)
ax[0].set_ylabel("time step t")
fig.colorbar(im, ax=ax, label="slowdown")
savefig("q2a_heatmaps.png"); plt.show()
for t in (0, 5, 10, 20):
    print(f"t={t:2d}: true peak at sensor {int(true_seq[t].argmax()):2d} (height {true_seq[t].max():.3f}) | "
          f"layer peak at sensor {int(pred_seq[t].argmax()):2d} (height {pred_seq[t].max():.3f}, "
          f"total mass {pred_seq[t].sum():.3f})")

# %% [markdown]
# ### (b) Direction-aware weights and the equivalent circular 1-D convolution
# $a=0,\ w_-=0,\ w_+=1$ gives $\hat x_i(t+1)=x_{i+1}(t)$ exactly. As a (cross-correlation) convolution the kernel
# multiplying $[x_{i-1},x_i,x_{i+1}]$ is $[0,0,1]$ with **circular** padding of one sensor on each side.

# %%
xr = torch.randn(1, 1, Nring, dtype=torch.float64)
kernel = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
y_conv = F.conv1d(F.pad(xr, (1, 1), mode="circular"), kernel)
print("conv == np.roll(x,-1) on random input:", torch.allclose(y_conv, torch.roll(xr, -1, dims=-1)))

# %% [markdown]
# ### (c) Using history with equal neighbour weights: $a=0,\ b=1,\ c=-1$

# %%
def check_history_layer(direction, a=0.0, b=1.0, c=-1.0, trials=200):
    shift = -1 if direction == "backward" else +1        # backward: x_i(t+1)=x_{i+1}(t)
    errs = []
    for _ in range(trials):
        x_prev = np.random.randn(Nring)
        x_t = np.roll(x_prev, shift); x_next = np.roll(x_t, shift)
        pred = a * x_t + b * (np.roll(x_t, 1) + np.roll(x_t, -1)) + c * x_prev
        errs.append(np.abs(pred - x_next).max())
    return max(errs)

print("max |error| backward motion:", check_history_layer("backward"))
print("max |error| forward  motion:", check_history_layer("forward"))

# %% [markdown]
# ---
# # Question 3 — Dataset preparation and graph construction
# ### Q3(a) Extract San Diego (District 11), 2020–2021, resample 5 → 15 min
#
# The raw LargeST files `ca_his_raw_{year}.h5` are pandas *fixed-format* HDF5 tables of shape
# (timestamps × 8 600 sensors, float64; ~7.2 GB each). To avoid loading 7 GB per year we read them with
# `h5py` in row blocks and keep only the 716 San Diego columns.
#
# **Aggregation choice.** LargeST/PeMS `flow` is the number of vehicles counted in each 5-minute interval
# (summed over lanes), i.e. an *interval count*. We therefore **sum** three consecutive 5-min counts so the
# 15-min value is *vehicles per 15 minutes*. When 1–2 of the three 5-min readings are missing we use
# 3 × (mean of the available readings), which equals the sum when nothing is missing and is unbiased otherwise.
# An interval with all three readings missing is marked missing.

# %%
def load_meta():
    if SMOKE:
        rs = np.random.RandomState(0); n = 60
        return pd.DataFrame({"ID": np.arange(1000, 1000 + n), "Lat": 32.7 + rs.rand(n) * 0.4,
                             "Lng": -117.2 + rs.rand(n) * 0.3, "District": 11,
                             "County": "San Diego", "Fwy": rs.choice(["I-5 S", "I-805 N", "SR-163 S"], n),
                             "Lanes": 3, "Type": "Mainline", "Direction": "S", "ID2": np.arange(n)})
    return pd.read_csv(os.path.join(DATA_DIR, "ca_meta.csv"))


def read_sd_year(year, sd_ids):
    """Return (DatetimeIndex 5-min, float32 array [T5, N_sd]) for one year, columns in sd_ids order."""
    import h5py
    path = os.path.join(DATA_DIR, f"ca_his_raw_{year}.h5")
    with h5py.File(path, "r") as f:
        g = f["t"]
        cols = g["axis0"][:].astype(str)
        pos = {c: i for i, c in enumerate(cols)}
        col_idx = np.array([pos[str(s)] for s in sd_ids])
        ts = pd.to_datetime(g["axis1"][:])
        vals = g["block0_values"]
        out = np.empty((vals.shape[0], len(col_idx)), dtype=np.float32)
        step = 8192
        for r0 in range(0, vals.shape[0], step):
            out[r0:r0 + step] = vals[r0:r0 + step, :][:, col_idx]
    return ts, out


def synthetic_year(year, n):
    ts = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:55", freq="5min")
    rs = np.random.RandomState(year)
    tod = (ts.hour * 60 + ts.minute).values / 1440.0
    base = 150 + 120 * np.sin(2 * np.pi * (tod - 0.3))[:, None] * (0.5 + rs.rand(n))[None]
    x = (base + rs.randn(len(ts), n) * 10).clip(0).astype(np.float32)
    x[rs.rand(*x.shape) < 0.01] = np.nan
    x[5000:5100, 3] = 0.0                                   # a synthetic zero-flow fault
    x[:, 7] = np.nan                                        # a dead sensor
    return ts, x


t0 = time.time()
meta = load_meta()
sd_meta = meta[meta["District"] == CFG["district"]].copy()
sd_rows = sd_meta.index.values                    # row positions -> index into ca_rn_adj.npy
sd_ids = sd_meta["ID"].values
N = len(sd_ids)
print("San Diego sensors:", N)

ts5_list, x5_list = [], []
for y in CFG["years"]:
    ts_y, x_y = read_sd_year(y, sd_ids) if not SMOKE else synthetic_year(y, N)
    full = pd.date_range(f"{y}-01-01", f"{y}-12-31 23:55", freq="5min")
    if not ts_y.equals(full):                     # guarantee a regular 5-min grid
        x_y = pd.DataFrame(x_y, index=ts_y).reindex(full).values.astype(np.float32); ts_y = full
    ts5_list.append(ts_y); x5_list.append(x_y)
    print(f"{y}: {x_y.shape}, NaN fraction {np.isnan(x_y).mean():.4%}, zero fraction {(x_y == 0).mean():.4%}")
ts5 = ts5_list[0].append(ts5_list[1]); X5 = np.concatenate(x5_list)
del x5_list
LOAD_TIME = time.time() - t0
print(f"5-min tensor {X5.shape}, loaded in {LOAD_TIME:.1f}s")

# %%
# ---- 5 -> 15 min aggregation (sum of counts; rescaled mean when partially missing)
T15 = X5.shape[0] // 3
X5r = X5[:T15 * 3].reshape(T15, 3, N)
n_obs = np.isfinite(X5r).sum(1)
with np.errstate(invalid="ignore"):
    X15_agg = np.where(n_obs > 0, np.nanmean(X5r, axis=1) * 3.0, np.nan).astype(np.float32)
ts15 = ts5[::3][:T15]
MISS_5MIN = float(np.isnan(X5).mean())
PARTIAL_15 = float(((n_obs > 0) & (n_obs < 3)).mean())
print("15-min tensor:", X15_agg.shape, ts15[0], "->", ts15[-1])
print(f"missing 5-min readings : {MISS_5MIN:.4%}")
print(f"15-min with partial data (1-2 of 3 present): {PARTIAL_15:.4%}")
print(f"15-min fully missing (NaN): {np.isnan(X15_agg).mean():.4%}")
del X5r, X5

# %% [markdown]
# **Zero flows.** A 15-minute total of zero vehicles on a freeway *mainline* detector is physically implausible for long,
# so we inspect when zeros occur and how long zero runs last. Runs of **≥ 4 consecutive zero 15-min totals (≥ 1 h)** are
# treated as detector faults and set to missing; isolated short zeros are kept as genuine (possible at night or during closures).

# %%
def run_lengths(mask_col):
    d = np.diff(np.concatenate([[0], mask_col.astype(np.int8), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s, e

Z = (X15_agg == 0)
zero_runs = []
fault = np.zeros_like(Z)
for j in range(N):
    if Z[:, j].any():
        s, e = run_lengths(Z[:, j])
        for a_, b_ in zip(s, e):
            zero_runs.append(b_ - a_)
            if b_ - a_ >= CFG["zero_run_steps"]:
                fault[a_:b_, j] = True
zero_runs = np.array(zero_runs) if zero_runs else np.zeros(0, int)
ZERO_STATS = {"zero_15min_frac": float(Z.mean()), "zero_runs": int(len(zero_runs)),
              "zero_values_in_runs_ge_1h": int(fault.sum()), "zero_values_short": int(Z.sum() - fault.sum()),
              "sensors_with_fault_runs": int(fault.any(0).sum()),
              "zeros_at_night_frac": float(Z[(ts15.hour < 5)].sum() / max(Z.sum(), 1))}
for k, v in ZERO_STATS.items():
    print(f"{k:>28}: {v}")
X15_raw = np.where(fault, np.nan, X15_agg).astype(np.float32)
OBS = np.isfinite(X15_raw)                     # True where a real, plausible 15-min measurement exists
miss_sensor = 1 - OBS.mean(0)
MISS_15 = float((~OBS).mean())
print(f"15-min missing after fault removal: {MISS_15:.4%}")
print(f"per-sensor missing rate: median {np.median(miss_sensor):.3%}, max {miss_sensor.max():.3%}, "
      f"sensors >10% missing: {(miss_sensor > 0.10).sum()}, sensors 100% missing: {(miss_sensor == 1).sum()}")

# %% [markdown]
# **Missing-value handling.**
# 1. A boolean mask `OBS` records which 15-min values were really observed (and plausible). The loss and **all metrics are
#    computed only on observed targets**, so imputed values are never treated as ground truth.
# 2. Model *inputs* are imputed **causally** — using only the past or training-set statistics, never later readings that may be
#    targets: gaps of ≤ 1 h (4 steps) are **forward-filled** with the last observed value; longer gaps (and gaps at the very
#    start) are filled with the sensor's **2020 time-of-week mean profile**; any residue falls back to the sensor's 2020 mean
#    and then the global 2020 mean. No validation/test information is used.

# %%
def split_bounds(name):
    s, e = CFG["split"][name]
    return int(np.searchsorted(ts15, pd.Timestamp(s))), int(np.searchsorted(ts15, pd.Timestamp(e), side="right")) - 1

TR, VA, TE = split_bounds("train"), split_bounds("val"), split_bounds("test")
print("index bounds  train", TR, " val", VA, " test", TE)
TOW = (ts15.dayofweek * 96 + ts15.hour * 4 + ts15.minute // 15).values   # 0..671


def fill_missing_causal(X, obs, train_end, max_gap):
    X = X.copy()
    T, n = X.shape
    tr = slice(0, train_end + 1)
    Xtr = np.where(obs[tr], X[tr], np.nan)
    df_tr = pd.DataFrame(Xtr); df_tr["tow"] = TOW[tr]
    prof = np.full((672, n), np.nan, dtype=np.float32)
    g = df_tr.groupby("tow").mean()
    prof[g.index.values] = g.values
    sensor_mean = np.nanmean(Xtr, axis=0)
    sensor_mean = np.where(np.isfinite(sensor_mean), sensor_mean, np.nanmean(sensor_mean))
    prof = np.where(np.isfinite(prof), prof, sensor_mean[None])
    n_ffill = n_prof = 0
    for j in range(n):
        m = ~obs[:, j]
        if not m.any():
            continue
        s_, e_ = run_lengths(m)
        for s, e in zip(s_, e_):
            L = e - s
            if L <= max_gap and s > 0:                    # last observation carried forward (past only)
                X[s:e, j] = X[s - 1, j]; n_ffill += L
            else:                                         # training-period time-of-week profile
                X[s:e, j] = prof[TOW[s:e], j]; n_prof += L
    return X, prof, n_ffill, n_prof

t0 = time.time()
X15, TOW_PROFILE, n_ffill, n_prof = fill_missing_causal(X15_raw, OBS, TR[1], CFG["short_gap_steps"])
print(f"inputs: {n_ffill} values forward-filled, {n_prof} from the 2020 time-of-week profile "
      f"({time.time() - t0:.1f}s); remaining NaN: {np.isnan(X15).sum()}")

# %%
# ---- Plot: three sensors over one week (+ missing / zero-fault overview)
week = (ts15 >= "2020-02-03") & (ts15 < "2020-02-10")          # Mon 3 – Sun 9 Feb 2020
mean_flow = np.nanmean(np.where(OBS, X15_raw, np.nan)[TR[0]:TR[1] + 1], 0)
ok_sensors = np.where(miss_sensor < 0.02)[0]
order = ok_sensors[np.argsort(mean_flow[ok_sensors])]
PLOT_SENSORS = [int(order[int(q * (len(order) - 1))]) for q in (0.95, 0.5, 0.15)]
fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for a_, s in zip(ax, PLOT_SENSORS):
    a_.plot(ts15[week], X15_raw[week, s], lw=1.1, label="15-min flow (observed)")
    miss = week & ~OBS[:, s]
    if miss.any():
        a_.scatter(ts15[miss], X15[miss, s], s=8, c="r", label="imputed")
    r = sd_meta.iloc[s]
    a_.set_title(f"sensor idx {s} (ID {r.ID}, {r.Fwy}, {r.Lanes} lanes)", fontsize=9)
    a_.set_ylabel("veh / 15 min")
ax[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Q3(a) Traffic flow for three San Diego sensors, 3–9 Feb 2020 (15-min sums)")
savefig("q3a_three_sensors_week.png"); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
ax[0].hist(miss_sensor * 100, bins=50, color="tab:gray"); ax[0].set_xlabel("missing rate per sensor (%)")
ax[0].set_ylabel("# sensors"); ax[0].set_yscale("log"); ax[0].set_title("per-sensor missing rate", fontsize=9)
daily = pd.Series((~OBS).mean(1), index=ts15).resample("D").mean() * 100
ax[1].plot(daily.index, daily.values, lw=1); ax[1].set_ylabel("missing (%) per day")
ax[1].set_title("daily missing rate (after fault removal)", fontsize=9)
hours = np.arange(24)
z_short = np.array([(Z & ~fault)[ts15.hour == h].sum() for h in hours])
z_fault = np.array([fault[ts15.hour == h].sum() for h in hours])
ax[2].bar(hours, z_short, label="short zero runs (kept)", color="tab:blue")
ax[2].bar(hours, z_fault, bottom=z_short, label="runs ≥ 1 h (fault → missing)", color="tab:red")
ax[2].set_xlabel("hour of day"); ax[2].set_ylabel("# zero 15-min totals"); ax[2].legend(fontsize=8)
ax[2].set_title("zero-flow intervals by hour", fontsize=9)
savefig("q3a_missing.png"); plt.show()

# %% [markdown]
# ### Q3(b) Road-distance adjacency for the 716 sensors
# `ca_rn_adj.npy` (8 600 × 8 600) is indexed by the row order of `ca_meta.csv`; the h5 columns use the same IDs,
# so `A_SD = A[sd_rows][:, sd_rows]` matches the column order of the traffic tensor.

# %%
if SMOKE:
    rs = np.random.RandomState(1)
    A_SD = (rs.rand(N, N) < 0.05) * rs.rand(N, N); np.fill_diagonal(A_SD, 1.0)
else:
    adj_full = np.load(os.path.join(DATA_DIR, "ca_rn_adj.npy"), mmap_mode="r")
    A_SD = np.asarray(adj_full[np.ix_(sd_rows, sd_rows)], dtype=np.float64)
    del adj_full
A_SD = A_SD.astype(np.float32)
offdiag = ~np.eye(N, dtype=bool)
nz = (A_SD != 0)
Abin = nz & offdiag
graph_stats = {
    "nodes": int(N),
    "nonzero_entries_incl_self_loops": int(nz.sum()),
    "self_loops": int(np.diag(nz).sum()),
    "directed_edges_excl_self": int(Abin.sum()),
    "undirected_pairs": int(np.triu(Abin | Abin.T).sum()),
    "symmetric_exact": bool(np.array_equal(A_SD, A_SD.T)),
    "symmetric_pattern": bool(np.array_equal(nz, nz.T)),
    "one_direction_only_pairs": int(np.triu(Abin != Abin.T).sum()),
    "weight_min_nonzero": float(A_SD[Abin].min()) if Abin.any() else None,
    "weight_max_offdiag": float(A_SD[offdiag].max()),
    "density_offdiag": float(Abin.sum() / (N * (N - 1))),
}
OUT_DEG = Abin.sum(1); IN_DEG = Abin.sum(0)
graph_stats.update(out_degree_mean=float(OUT_DEG.mean()), out_degree_median=float(np.median(OUT_DEG)),
                   out_degree_max=int(OUT_DEG.max()), isolated_nodes=int(((OUT_DEG + IN_DEG) == 0).sum()))
for k, v in graph_stats.items():
    print(f"{k:>34}: {v}")

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
im = ax[0].imshow(np.where(A_SD > 0, A_SD, np.nan), cmap="viridis", interpolation="nearest")
ax[0].set_title("Q3(b) $A_{SD}$ (716×716, zeros blank)"); ax[0].grid(False)
fig.colorbar(im, ax=ax[0], fraction=0.046)
ax[1].imshow(np.where(A_SD[:120, :120] > 0, A_SD[:120, :120], np.nan), cmap="viridis", interpolation="nearest")
ax[1].set_title("zoom: first 120 sensors"); ax[1].grid(False)
bins = np.arange(0, max(OUT_DEG.max(), IN_DEG.max()) + 2) - 0.5
ax[2].hist(OUT_DEG, bins=bins, alpha=0.6, label=f"out-degree (mean {OUT_DEG.mean():.1f})")
ax[2].hist(IN_DEG, bins=bins, alpha=0.6, label=f"in-degree (mean {IN_DEG.mean():.1f})")
ax[2].set_xlabel("# neighbours (excluding self)"); ax[2].set_ylabel("# sensors"); ax[2].legend()
ax[2].set_title("neighbour-count distribution")
savefig("q3b_adjacency.png"); plt.show()

# %% [markdown]
# **Reliable message sources.** Sensors with < 50 % observed 2020 data have (almost) no real training signal: their inputs are
# mostly imputed constants or profiles. They remain forecasting *targets* (masked), but they are **not used as message sources**
# in any graph — their columns are removed from the fixed graph before normalisation and they cannot be selected by the adaptive
# graph — so no model can exploit imputed artefacts as "neighbours".

# %%
train_obs_frac = OBS[TR[0]:TR[1] + 1].mean(0)
SRC_OK = train_obs_frac >= CFG["src_min_obs"]
print(f"reliable source sensors: {SRC_OK.sum()} / {N}  (excluded as sources: {(~SRC_OK).sum()}, "
      f"of which with no 2020 observation at all: {(train_obs_frac == 0).sum()})")
print(f"road edges removed because their source is unreliable: {int(Abin[:, ~SRC_OK].sum())} of {int(Abin.sum())}")

# %% [markdown]
# ### Q3(c) One-hop and two-hop neighbourhoods of a selected sensor

# %%
two_hop_reach = ((Abin.astype(np.int32) @ Abin.astype(np.int32)) > 0) | Abin
np.fill_diagonal(two_hop_reach, False)
cand = np.where((OUT_DEG == int(np.median(OUT_DEG))) & SRC_OK)[0]
SEL = int(cand[0]) if len(cand) else int(np.argsort(OUT_DEG)[N // 2])
one_hop = np.where(Abin[SEL])[0]
two_hop = np.where(two_hop_reach[SEL])[0]
two_only = np.setdiff1d(two_hop, one_hop)
r = sd_meta.iloc[SEL]
print(f"selected sensor idx {SEL} (ID {r.ID}, {r.Fwy}, lat {r.Lat:.4f}, lng {r.Lng:.4f})")
print(f"1-hop neighbours ({len(one_hop)}):", [int(sd_ids[j]) for j in one_hop])
print(f"2-hop neighbourhood (<=2 edges) size {len(two_hop)}; reachable only in exactly 2 hops: {len(two_only)}")
CORR = np.corrcoef(X15[TR[0]:TR[1] + 1].T)          # training-period correlations only
nbr_table = pd.DataFrame({"idx": np.r_[one_hop, two_only], "hop": [1] * len(one_hop) + [2] * len(two_only)})
nbr_table["ID"] = sd_ids[nbr_table.idx]; nbr_table["Fwy"] = sd_meta.Fwy.values[nbr_table.idx]
nbr_table["A_sel_j"] = A_SD[SEL, nbr_table.idx].round(3)
nbr_table["train_corr"] = CORR[SEL, nbr_table.idx].round(3)
nbr_table["reliable"] = SRC_OK[nbr_table.idx]
print(nbr_table.to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(sd_meta.Lng, sd_meta.Lat, s=4, c="lightgray", label="all SD sensors")
ax.scatter(sd_meta.Lng.values[two_only], sd_meta.Lat.values[two_only], s=25, c="tab:orange", label="2-hop only")
ax.scatter(sd_meta.Lng.values[one_hop], sd_meta.Lat.values[one_hop], s=30, c="tab:blue", label="1-hop")
ax.scatter([r.Lng], [r.Lat], s=120, marker="*", c="red", label=f"selected (idx {SEL})")
pad = 0.06
lat_nb = sd_meta.Lat.values[two_hop.tolist() + [SEL]]; lng_nb = sd_meta.Lng.values[two_hop.tolist() + [SEL]]
ax.set_xlim(lng_nb.min() - pad, lng_nb.max() + pad); ax.set_ylim(lat_nb.min() - pad, lat_nb.max() + pad)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(fontsize=8)
ax.set_title("Q3(c) one- and two-hop neighbourhood")
savefig("q3c_neighbourhood.png"); plt.show()

# %% [markdown]
# ---
# # Question 4 — Spatio-temporal forecasting
# ### Q4(a) Windows, splits and normalisation
# Sample $t$ uses $X_t=[x_{t-7},\dots,x_t]$ and $Y_t=[x_{t+1},\dots,x_{t+4}]$. A window belongs to a split only if
# **all 12 time steps** lie inside that split, so no window crosses a boundary. Z-score statistics
# (one global mean/std) come from **observed 2020 values only**.

# %%
def window_index(bounds):
    s, e = bounds
    return np.arange(s + CFG["t_in"] - 1, e - CFG["t_out"] + 1)   # t with t-7 >= s and t+4 <= e

IDX = {k: window_index(b) for k, b in [("train", TR), ("val", VA), ("test", TE)]}
train_obs = X15_raw[TR[0]:TR[1] + 1][OBS[TR[0]:TR[1] + 1]]
MU, SD = float(train_obs.mean()), float(train_obs.std())
print(f"train mean {MU:.3f}, std {SD:.3f} (veh/15 min)")
for k, v in IDX.items():
    print(f"{k:>5}: {len(v):6d} windows, first target time {ts15[v[0] + 1]}, last target time {ts15[v[-1] + 4]}")
assert IDX["train"][-1] + CFG["t_out"] <= TR[1] and IDX["val"][0] - CFG["t_in"] + 1 >= VA[0]
assert IDX["val"][-1] + CFG["t_out"] <= VA[1] and IDX["test"][0] - CFG["t_in"] + 1 >= TE[0]

# %% [markdown]
# ### Q4(b) Model, Q5 graphs and the training library
# The model/training code lives in `stgnn_lib.py` (written by the next cell, also in the repository):
# * **Temporal encoder** (shared by all sensors): `Conv1d(1→d, k=3)` → ReLU → `Conv1d(d→d, k=3, dilation=2)` → ReLU → flatten →
#   `Linear(2d→d)` → ReLU, giving $Z\in\mathbb R^{B\times N\times d}$, $d=32$.
# * **Graph convolution** $H=\mathrm{ReLU}(\hat A Z W_g)$ via `einsum('nm,bmd->bnd')`. Main models: 2 layers with residual
#   connections $H\leftarrow H+\mathrm{ReLU}(\hat A H W)$; the ablation `fixed_spec` uses exactly the handout form (1 layer, no residual).
# * **Head**: per-node MLP `Linear(d→d)` → ReLU → `Linear(d→4)`.
# * **Fixed graph**: $\hat A=D^{-1}\tilde A$ (directed random-walk normalisation); ablation `fixed_bi` uses $D^{-1}(\tilde A+\tilde A^\top)$.
# * **Adaptive graph**: $\mathrm{Softmax}_{row}(\mathrm{TopK}_{20}(\mathrm{ReLU}(E_1E_2^\top)))$, $E_1,E_2\in\mathbb R^{716\times10}$.
# * **Hybrid**: $\alpha\hat A_{fixed}+(1-\alpha)A_{adaptive}$, $\alpha=\sigmoid(a)$ learned.
# * **Ablations for node identity**: `none_emb` / `fixed_emb` add a learnable per-sensor embedding ($716\times20$ → `Linear(20→d)`,
#   ≈ the same 14 k extra parameters as the adaptive graph) to $Z$, separating *node-specific capacity* from *learned structure*.

# %%
LIB_SRC = r'''"""STGNN library for AI651 PA3 (models, training, evaluation, multi-GPU worker).

The notebook writes this file next to itself, imports it for parameter counts, and launches one worker
process per GPU:  python stgnn_lib.py --prep prep.npz --runs fixed:42,adaptive:42 --out runs/
Each worker claims runs from a shared list via exclusive lock files, so the GPUs stay busy and every run
gets a dedicated GPU (fair training-time measurements).
"""
import os, sys, json, time, math, random, argparse

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------- configurations
# adj: which graph; layers/residual: graph-conv stack; node_emb: per-node embedding dim added to Z (0 = none)
CONFIGS = {
    "none":       dict(label="Temporal-only",            adj="none",     layers=0, residual=True,  node_emb=0),
    "fixed":      dict(label="STGNN, fixed",             adj="fixed",    layers=2, residual=True,  node_emb=0),
    "adaptive":   dict(label="STGNN, adaptive",          adj="adaptive", layers=2, residual=True,  node_emb=0),
    "hybrid":     dict(label="STGNN, hybrid",            adj="hybrid",   layers=2, residual=True,  node_emb=0),
    # ---- ablations
    "fixed_spec": dict(label="Fixed, 1 GC layer, no residual (handout form)", adj="fixed", layers=1, residual=False, node_emb=0),
    "fixed_bi":   dict(label="Fixed, bidirectional graph", adj="fixed_bi", layers=2, residual=True, node_emb=0),
    "none_emb":   dict(label="Temporal-only + node embeddings", adj="none", layers=0, residual=True, node_emb=20),
    "fixed_emb":  dict(label="Fixed + node embeddings",  adj="fixed",    layers=2, residual=True,  node_emb=20),
}
MAIN = ["none", "fixed", "adaptive", "hybrid"]
ABLATIONS = ["fixed_spec", "fixed_bi", "none_emb", "fixed_emb"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------------------- graph helpers
def normalize_fixed(A, src_ok=None, bidirectional=False):
    """Directed random-walk normalisation  A_hat = D^{-1} A_tilde.
    A_tilde = A (or A + A^T if bidirectional) with unit self-loops; columns of unreliable sensors
    (no usable training signal) are removed as message SOURCES (their own self-loop is kept)."""
    At = A + A.T if bidirectional else A.copy()
    At = At.astype(np.float64)
    if src_ok is not None:
        At[:, ~src_ok] = 0.0
    np.fill_diagonal(At, 1.0)
    return (At / At.sum(1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------------------- model
class TemporalEncoder(nn.Module):
    """Shared per-sensor temporal conv: Conv1d(1->d,k3) -> ReLU -> Conv1d(d->d,k3,dil2) -> ReLU -> Linear -> ReLU."""
    def __init__(self, d, t_in):
        super().__init__()
        self.c1 = nn.Conv1d(1, d, 3)
        self.c2 = nn.Conv1d(d, d, 3, dilation=2)
        self.proj = nn.Linear(d * (t_in - 6), d)

    def forward(self, x):                       # x: [B, T, N]
        B, T, N = x.shape
        h = x.permute(0, 2, 1).reshape(B * N, 1, T)
        h = F.relu(self.c2(F.relu(self.c1(h))))
        return F.relu(self.proj(h.flatten(1))).view(B, N, -1)       # Z: [B, N, d]


class GraphConv(nn.Module):
    """ReLU(A Z W): row i of A aggregates over columns j, independently for every batch item."""
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d)

    def forward(self, Z, A):
        return F.relu(torch.einsum("nm,bmd->bnd", A, self.W(Z)))


class FixedAdj(nn.Module):
    def __init__(self, A_hat):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(A_hat), persistent=False)   # never trained

    def forward(self):
        return self.A


class AdaptiveAdj(nn.Module):
    """A_adp = Softmax_row(TopK_k(ReLU(E1 E2^T))). Top-k is applied to the logits BEFORE the softmax,
    so every row is a distribution over its k kept neighbours. Unreliable sensors cannot be selected
    as sources (except by themselves)."""
    def __init__(self, n, de, topk, src_ok=None):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(n, de))
        self.E2 = nn.Parameter(torch.randn(n, de))
        self.topk = topk
        if src_ok is None:
            src_ok = np.ones(n, dtype=bool)
        block = ~torch.as_tensor(src_ok)[None, :].expand(n, n).clone()
        block.fill_diagonal_(False)
        self.register_buffer("block", block, persistent=False)

    def forward(self):
        logits = F.relu(self.E1 @ self.E2.T).masked_fill(self.block, float("-inf"))
        if self.topk and self.topk < logits.shape[1]:
            vals, idx = logits.topk(self.topk, dim=1)
            logits = torch.full_like(logits, float("-inf")).scatter(1, idx, vals)
        return torch.softmax(logits, dim=1)


class HybridAdj(nn.Module):
    """alpha * A_fixed + (1 - alpha) * A_adaptive, both row-stochastic; alpha = sigmoid(a) is learned."""
    def __init__(self, A_hat, n, de, topk, src_ok, alpha_init=0.5):
        super().__init__()
        self.fixed = FixedAdj(A_hat)
        self.adaptive = AdaptiveAdj(n, de, topk, src_ok)
        self.a = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    @property
    def alpha(self):
        return torch.sigmoid(self.a)

    def forward(self):
        return self.alpha * self.fixed() + (1 - self.alpha) * self.adaptive()


class STGNN(nn.Module):
    """Temporal encoder -> [optional node embedding] -> L graph-conv layers (optionally residual) -> MLP head."""
    def __init__(self, n, d, t_in, t_out, adj=None, layers=2, residual=True, node_emb=0):
        super().__init__()
        self.encoder = TemporalEncoder(d, t_in)
        self.node_emb = nn.Parameter(torch.randn(n, node_emb)) if node_emb else None
        self.emb_proj = nn.Linear(node_emb, d) if node_emb else None
        self.adj = adj
        self.residual = residual
        self.gcns = nn.ModuleList([GraphConv(d) for _ in range(layers if adj is not None else 0)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, t_out))

    def forward(self, x):                        # [B, T_in, N] -> [B, T_out, N]
        H = self.encoder(x)
        if self.node_emb is not None:
            H = H + self.emb_proj(self.node_emb)[None]
        if self.adj is not None:
            A = self.adj()
            for g in self.gcns:
                H = H + g(H, A) if self.residual else g(H, A)
        return self.head(H).permute(0, 2, 1)


def build_model(name, cfg, graphs, n, device="cpu", seed=None):
    if seed is not None:
        set_seed(seed)
    c = CONFIGS[name]
    adj = {"none": lambda: None,
           "fixed": lambda: FixedAdj(graphs["A_fixed"]),
           "fixed_bi": lambda: FixedAdj(graphs["A_bi"]),
           "adaptive": lambda: AdaptiveAdj(n, cfg["emb_dim"], cfg["topk"], graphs["src_ok"]),
           "hybrid": lambda: HybridAdj(graphs["A_fixed"], n, cfg["emb_dim"], cfg["topk"], graphs["src_ok"],
                                       cfg["alpha_init"])}[c["adj"]]()
    return STGNN(n, cfg["hidden"], cfg["t_in"], cfg["t_out"], adj, c["layers"], c["residual"], c["node_emb"]).to(device)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------- data / metrics
class Data:
    def __init__(self, prep, device):
        self.device = device
        self.data = torch.tensor(prep["data_norm"], device=device)
        self.raw = torch.tensor(prep["raw"], device=device)
        self.mask = torch.tensor(prep["mask"], device=device)
        self.mu, self.sd = float(prep["mu"]), float(prep["sd"])
        self.idx = {k: prep[f"idx_{k}"] for k in ["train", "val", "test"]}
        t_in, t_out = int(prep["t_in"]), int(prep["t_out"])
        self.off_in = torch.arange(-t_in + 1, 1, device=device)
        self.off_out = torch.arange(1, t_out + 1, device=device)
        self.t_out = t_out

    def batch(self, t_idx):
        t_idx = torch.as_tensor(t_idx, device=self.device)
        yi = t_idx[:, None] + self.off_out
        return self.data[t_idx[:, None] + self.off_in], self.data[yi], self.raw[yi], self.mask[yi]


def masked_mae(pred, y, m):
    m = m.float()
    return (torch.abs(pred - y) * m).sum() / m.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, D, t_idx, bs=256, keep=None):
    model.eval()
    ae = torch.zeros(D.t_out, device=D.device, dtype=torch.float64); se = torch.zeros_like(ae); cnt = torch.zeros_like(ae)
    kept = []
    for i in range(0, len(t_idx), bs):
        x, _, y, m = D.batch(t_idx[i:i + bs])
        p = model(x) * D.sd + D.mu
        mf = m.double(); err = (p - y).double()
        ae += (err.abs() * mf).sum((0, 2)); se += (err ** 2 * mf).sum((0, 2)); cnt += mf.sum((0, 2))
        if keep is not None:
            kept.append(p[:, :, keep].float().cpu())
    out = {"MAE": float(ae.sum() / cnt.sum()), "RMSE": float(torch.sqrt(se.sum() / cnt.sum())),
           "MAE_h": (ae / cnt).cpu().tolist(), "RMSE_h": torch.sqrt(se / cnt).cpu().tolist()}
    return out, (torch.cat(kept).numpy() if keep is not None else None)


def train_run(name, seed, cfg, D, graphs, n, out_dir, keep=None, log=print):
    model = build_model(name, cfg, graphs, n, D.device, seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg["lr_factor"],
                                                       patience=cfg["lr_patience"], min_lr=cfg["min_lr"])
    gen = torch.Generator().manual_seed(seed)
    tr = D.idx["train"]
    best, best_state, best_ep, bad, hist = float("inf"), None, 0, 0, []
    sync = (lambda: torch.cuda.synchronize()) if D.device.type == "cuda" else (lambda: None)
    sync(); t0 = time.time()
    for ep in range(1, cfg["max_epochs"] + 1):
        model.train(); te = time.time(); run, nb = 0.0, 0
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for i in range(0, len(perm), cfg["batch_size"]):
            x, y, _, m = D.batch(perm[i:i + cfg["batch_size"]])
            loss = masked_mae(model(x), y, m)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]); opt.step()
            run += loss.item(); nb += 1
        val, _ = evaluate(model, D, D.idx["val"])
        sched.step(val["MAE"])
        sync()
        rec = {"epoch": ep, "train_mae": run / nb * D.sd, "val_mae": val["MAE"], "val_rmse": val["RMSE"],
               "lr": opt.param_groups[0]["lr"], "epoch_time_s": time.time() - te}
        if hasattr(model.adj, "alpha"):
            rec["alpha"] = float(model.adj.alpha)
        hist.append(rec)
        log(f"[{name}:{seed}] ep {ep:2d} train {rec['train_mae']:.3f} val {val['MAE']:.3f}/{val['RMSE']:.3f} "
            f"lr {rec['lr']:.1e} {rec['epoch_time_s']:.1f}s" + (f" alpha {rec['alpha']:.3f}" if "alpha" in rec else ""))
        if val["MAE"] < best - 1e-4:
            best, bad, best_ep = val["MAE"], 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    res = {"config": name, "seed": seed, "label": CONFIGS[name]["label"], "n_params": n_params(model),
           "train_time_s": train_time, "epochs_run": ep, "best_epoch": best_ep, "sec_per_epoch": train_time / ep,
           "stopped_early": ep < cfg["max_epochs"], "history": hist,
           "device": torch.cuda.get_device_name(D.device) if D.device.type == "cuda" else "cpu"}
    res["val"], _ = evaluate(model, D, D.idx["val"])
    res["test"], kept = evaluate(model, D, D.idx["test"], keep=keep)
    tag = f"{name}_s{seed}"
    if hasattr(model.adj, "alpha"):
        res["alpha_final"] = float(model.adj.alpha)
    torch.save(model.state_dict(), os.path.join(out_dir, f"model_{tag}.pt"))
    with torch.no_grad():
        if CONFIGS[name]["adj"] in ("adaptive", "hybrid"):
            np.save(os.path.join(out_dir, f"A_{tag}.npy"), model.adj().cpu().numpy().astype(np.float32))
        if CONFIGS[name]["adj"] == "hybrid":
            np.save(os.path.join(out_dir, f"A_{tag}_adaptive_part.npy"),
                    model.adj.adaptive().cpu().numpy().astype(np.float32))
    if keep is not None:
        np.save(os.path.join(out_dir, f"pred_{tag}.npy"), kept.astype(np.float32))
    with open(os.path.join(out_dir, f"run_{tag}.json"), "w") as f:
        json.dump(res, f)
    return res


# --------------------------------------------------------------------------------------- worker entry point
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--runs", required=True, help="comma list name:seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", default="w0")
    a = ap.parse_args()
    prep = np.load(a.prep, allow_pickle=False)
    cfg = json.loads(str(prep["cfg_json"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = Data(prep, device)
    graphs = {"A_fixed": prep["A_fixed"], "A_bi": prep["A_bi"], "src_ok": prep["src_ok"]}
    n = prep["A_fixed"].shape[0]
    keep = prep["keep"].tolist()
    claims = os.path.join(a.out, "claims"); os.makedirs(claims, exist_ok=True)
    log = lambda s: print(f"{time.strftime('%H:%M:%S')} {a.worker} {s}", flush=True)
    log(f"device {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    for item in a.runs.split(","):
        name, seed = item.split(":"); seed = int(seed)
        try:   # exclusive claim -> dynamic load balancing across workers
            os.close(os.open(os.path.join(claims, f"{name}_s{seed}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        r = train_run(name, seed, cfg, D, graphs, n, a.out, keep=keep if seed == cfg["seeds"][0] else None, log=log)
        log(f"DONE {name}:{seed} test MAE {r['test']['MAE']:.3f} RMSE {r['test']['RMSE']:.3f} "
            f"time {r['train_time_s']:.0f}s epochs {r['epochs_run']} (best {r['best_epoch']})")


if __name__ == "__main__":
    main()
'''
LIB_PATH = os.path.join(WORK, "stgnn_lib.py")
with open(LIB_PATH, "w", encoding="utf-8") as f:
    f.write(LIB_SRC)
sys.path.insert(0, WORK)
import importlib, stgnn_lib
importlib.reload(stgnn_lib)
from stgnn_lib import CONFIGS, MAIN, ABLATIONS, normalize_fixed, build_model, n_params

A_FIXED = normalize_fixed(A_SD, SRC_OK)
A_BI = normalize_fixed(A_SD, SRC_OK, bidirectional=True)
GRAPHS = {"A_fixed": A_FIXED, "A_bi": A_BI, "src_ok": SRC_OK}
PARAMS = {k: n_params(build_model(k, CFG, GRAPHS, N, "cpu", SEED)) for k in CONFIGS}
print(pd.DataFrame({"label": [CONFIGS[k]["label"] for k in CONFIGS], "trainable params": list(PARAMS.values())},
                   index=list(CONFIGS)).to_string())

# %% [markdown]
# ### Training protocol (identical for every configuration and seed)
# Adam (lr 1e-3) with ReduceLROnPlateau (×0.5 after 3 epochs without validation improvement, min 1e-5), batch 64,
# masked-MAE loss on normalised values, gradient clipping 5, up to 80 epochs with early stopping on validation MAE
# (patience 10) and restoration of the best weights. Seeds 42, 43, 44. Metrics in **original units (vehicles / 15 min)** on
# observed targets only. Training time = wall-clock time of the training loop incl. per-epoch validation, on a dedicated GPU.

# %%
PREP = os.path.join(TMP_DIR, "prep.npz")
np.savez(PREP, data_norm=((X15 - MU) / SD).astype(np.float32), raw=np.nan_to_num(X15_raw).astype(np.float32),
         mask=OBS, idx_train=IDX["train"], idx_val=IDX["val"], idx_test=IDX["test"], mu=MU, sd=SD,
         t_in=CFG["t_in"], t_out=CFG["t_out"], A_fixed=A_FIXED, A_bi=A_BI, src_ok=SRC_OK,
         keep=np.array(PLOT_SENSORS), cfg_json=json.dumps(CFG))
MEM_PREP_GB = peak_rss_gb()
print(f"prepared tensors saved; peak RAM of the notebook process so far: {MEM_PREP_GB:.2f} GB")

# %%
RUNS = [f"{c}:{s}" for s in CFG["seeds"] for c in MAIN + ABLATIONS]
n_workers = max(1, torch.cuda.device_count())
shutil.rmtree(os.path.join(RUN_DIR, "claims"), ignore_errors=True)
procs, logs = [], []
t_all = time.time()
for w in range(n_workers):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w) if torch.cuda.is_available() else "", OMP_NUM_THREADS="2")
    lf = open(os.path.join(RUN_DIR, f"worker{w}.log"), "w")
    procs.append(subprocess.Popen([sys.executable, LIB_PATH, "--prep", PREP, "--runs", ",".join(RUNS),
                                   "--out", RUN_DIR, "--worker", f"gpu{w}"], stdout=lf, stderr=subprocess.STDOUT, env=env))
    logs.append(lf)
print(f"launched {n_workers} worker(s) for {len(RUNS)} runs")
done_seen = set()
while any(p.poll() is None for p in procs):
    time.sleep(30 if not SMOKE else 5)
    done = {f for f in os.listdir(RUN_DIR) if f.startswith("run_")}
    for f in sorted(done - done_seen):
        r = json.load(open(os.path.join(RUN_DIR, f)))
        print(f"[{(time.time() - t_all) / 60:6.1f} min] {r['config']:>10}:{r['seed']} test MAE {r['test']['MAE']:.3f} "
              f"RMSE {r['test']['RMSE']:.3f} | train {r['train_time_s']:.0f}s, {r['epochs_run']} ep (best {r['best_epoch']}) on {r['device']}")
    done_seen = done
for lf in logs:
    lf.close()
for w, p in enumerate(procs):
    if p.returncode != 0:
        print(open(os.path.join(RUN_DIR, f"worker{w}.log")).read()[-5000:])
        raise RuntimeError(f"worker {w} failed with code {p.returncode}")
TOTAL_TRAIN_WALL = time.time() - t_all
print(f"all runs finished in {TOTAL_TRAIN_WALL / 60:.1f} min wall-clock")

# %%
RES = {}
for f in os.listdir(RUN_DIR):
    if f.startswith("run_"):
        r = json.load(open(os.path.join(RUN_DIR, f)))
        RES[(r["config"], r["seed"])] = r
assert len(RES) == len(RUNS), f"missing runs: {set(RUNS) - {f'{c}:{s}' for c, s in RES}}"
print("example log (adaptive, first seed):")
for h in RES[("adaptive", SEED)]["history"]:
    print(f"  ep {h['epoch']:2d} train {h['train_mae']:.3f} val {h['val_mae']:.3f} lr {h['lr']:.1e}")

# %% [markdown]
# ---
# # Question 6(a) — Forecasting comparison (mean ± std over 3 seeds)

# %%
def agg(cfg, key):
    vals = np.array([key(RES[(cfg, s)]) for s in CFG["seeds"]])
    return vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0

def pm(m, s, nd=2):
    return f"{m:.{nd}f} ± {s:.{nd}f}"

rows = []
for c in MAIN + ABLATIONS:
    row = {"config": c, "label": CONFIGS[c]["label"], "params": PARAMS[c]}
    for k, fn in [("val_MAE", lambda r: r["val"]["MAE"]), ("val_RMSE", lambda r: r["val"]["RMSE"]),
                  ("test_MAE", lambda r: r["test"]["MAE"]), ("test_RMSE", lambda r: r["test"]["RMSE"]),
                  ("train_time_s", lambda r: r["train_time_s"]), ("epochs", lambda r: r["epochs_run"]),
                  ("best_epoch", lambda r: r["best_epoch"]), ("s_per_epoch", lambda r: r["sec_per_epoch"])]:
        row[k + "_mean"], row[k + "_std"] = agg(c, fn)
    for h in range(CFG["t_out"]):
        row[f"test_MAE_{15 * (h + 1)}m_mean"], row[f"test_MAE_{15 * (h + 1)}m_std"] = agg(c, lambda r: r["test"]["MAE_h"][h])
        row[f"test_RMSE_{15 * (h + 1)}m_mean"], row[f"test_RMSE_{15 * (h + 1)}m_std"] = agg(c, lambda r: r["test"]["RMSE_h"][h])
    row["all_stopped_early"] = all(RES[(c, s)]["stopped_early"] for s in CFG["seeds"])
    if c == "hybrid":
        row["alpha_mean"], row["alpha_std"] = agg(c, lambda r: r["alpha_final"])
    rows.append(row)
SUMMARY = pd.DataFrame(rows)
SUMMARY.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)
per_seed = pd.DataFrame([{"config": c, "seed": s, "val_MAE": r["val"]["MAE"], "val_RMSE": r["val"]["RMSE"],
                          "test_MAE": r["test"]["MAE"], "test_RMSE": r["test"]["RMSE"], "train_time_s": r["train_time_s"],
                          "epochs": r["epochs_run"], "best_epoch": r["best_epoch"], "stopped_early": r["stopped_early"],
                          "alpha": r.get("alpha_final"), "device": r["device"]} for (c, s), r in sorted(RES.items())])
per_seed.to_csv(os.path.join(OUT_DIR, "results_per_seed.csv"), index=False)

show = pd.DataFrame({"Model": SUMMARY.label, "Val MAE": [pm(a, b) for a, b in zip(SUMMARY.val_MAE_mean, SUMMARY.val_MAE_std)],
                     "Val RMSE": [pm(a, b) for a, b in zip(SUMMARY.val_RMSE_mean, SUMMARY.val_RMSE_std)],
                     "Test MAE": [pm(a, b) for a, b in zip(SUMMARY.test_MAE_mean, SUMMARY.test_MAE_std)],
                     "Test RMSE": [pm(a, b) for a, b in zip(SUMMARY.test_RMSE_mean, SUMMARY.test_RMSE_std)],
                     "Train time (s)": [pm(a, b, 0) for a, b in zip(SUMMARY.train_time_s_mean, SUMMARY.train_time_s_std)],
                     "Epochs": [pm(a, b, 1) for a, b in zip(SUMMARY.epochs_mean, SUMMARY.epochs_std)],
                     "early stop (all seeds)": SUMMARY.all_stopped_early, "params": SUMMARY.params})
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(show.to_string(index=False))
print("\nper-horizon test MAE (mean ± std):")
print(pd.DataFrame({"Model": SUMMARY.label, **{f"{m} min": [pm(a, b) for a, b in zip(SUMMARY[f"test_MAE_{m}m_mean"], SUMMARY[f"test_MAE_{m}m_std"])]
                                                for m in (15, 30, 45, 60)}}).to_string(index=False))
print("\nper-horizon test RMSE (mean ± std):")
print(pd.DataFrame({"Model": SUMMARY.label, **{f"{m} min": [pm(a, b) for a, b in zip(SUMMARY[f"test_RMSE_{m}m_mean"], SUMMARY[f"test_RMSE_{m}m_std"])]
                                                for m in (15, 30, 45, 60)}}).to_string(index=False))
if "alpha_mean" in SUMMARY:
    print("\nhybrid alpha per seed:", [round(RES[("hybrid", s)]["alpha_final"], 4) for s in CFG["seeds"]])

# %%
# Paired comparisons: same seed => same initialisation of the shared modules and same batch order.
def paired(a, b):
    d = np.array([RES[(a, s)]["test"]["MAE"] - RES[(b, s)]["test"]["MAE"] for s in CFG["seeds"]])
    wins = int((d < 0).sum())
    return d.mean(), d.std(ddof=1) if len(d) > 1 else 0.0, wins

PAIRS = [("fixed", "none"), ("adaptive", "fixed"), ("hybrid", "fixed"), ("hybrid", "adaptive"),
         ("fixed_spec", "fixed"), ("fixed_bi", "fixed"), ("none_emb", "none"), ("fixed_emb", "fixed"),
         ("adaptive", "fixed_emb"), ("adaptive", "none_emb")]
PAIRED = pd.DataFrame([{"A": a, "B": b, "mean dMAE (A-B)": d, "std": s, f"A better in /{len(CFG['seeds'])} seeds": w}
                       for a, b in PAIRS for d, s, w in [paired(a, b)]])
PAIRED.to_csv(os.path.join(OUT_DIR, "paired_comparisons.csv"), index=False)
print(PAIRED.round(3).to_string(index=False))

# naive reference points on the same test targets
def naive_metrics(kind):
    t_idx = IDX["test"]; abs_e = sq_e = cnt = 0.0
    for i in range(0, len(t_idx), 1024):
        ti = t_idx[i:i + 1024]
        yi = ti[:, None] + np.arange(1, CFG["t_out"] + 1)
        y, m = X15_raw[yi], OBS[yi]
        p = np.repeat(X15[ti][:, None], CFG["t_out"], 1) if kind == "last" else TOW_PROFILE[TOW[yi]]
        e = np.where(m, p - np.nan_to_num(y), 0)
        abs_e += np.abs(e).sum(); sq_e += (e ** 2).sum(); cnt += m.sum()
    return abs_e / cnt, math.sqrt(sq_e / cnt)

NAIVE = {k: naive_metrics(k) for k in ["last", "tow"]}
print("\nreference (test): last-value MAE/RMSE = %.3f / %.3f ; 2020 time-of-week profile MAE/RMSE = %.3f / %.3f"
      % (*NAIVE["last"], *NAIVE["tow"]))

# %%
colors = {"none": "tab:gray", "fixed": "tab:blue", "adaptive": "tab:orange", "hybrid": "tab:green"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for c in MAIN:
    h = pd.DataFrame(RES[(c, SEED)]["history"])
    ax[0].plot(h.epoch, h.train_mae, c=colors[c], ls="--", lw=1)
    ax[0].plot(h.epoch, h.val_mae, c=colors[c], label=CONFIGS[c]["label"])
    for a_, key in [(ax[1], "MAE"), (ax[2], "RMSE")]:
        m = SUMMARY.set_index("config").loc[c]
        mean = [m[f"test_{key}_{x}m_mean"] for x in (15, 30, 45, 60)]; sd = [m[f"test_{key}_{x}m_std"] for x in (15, 30, 45, 60)]
        a_.errorbar([15, 30, 45, 60], mean, yerr=sd, fmt="o-", c=colors[c], capsize=3)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MAE (veh/15 min)"); ax[0].set_title(f"train (dashed) / val (solid) MAE, seed {SEED}")
ax[0].legend(fontsize=8)
for a_, t_ in zip(ax[1:], ["test MAE per horizon (mean ± std, 3 seeds)", "test RMSE per horizon (mean ± std, 3 seeds)"]):
    a_.set_xlabel("horizon (min)"); a_.set_title(t_, fontsize=9); a_.set_xticks([15, 30, 45, 60])
savefig("q6a_curves_horizons.png"); plt.show()

# ablation bar chart
fig, ax = plt.subplots(figsize=(10, 3.8))
order_ = MAIN + ABLATIONS
m_ = SUMMARY.set_index("config").loc[order_]
ax.bar(range(len(order_)), m_.test_MAE_mean, yerr=m_.test_MAE_std, capsize=4,
       color=[colors.get(c, "tab:purple") for c in order_])
for i, c in enumerate(order_):
    for s in CFG["seeds"]:
        ax.scatter(i, RES[(c, s)]["test"]["MAE"], c="k", s=10, zorder=3)
ax.set_xticks(range(len(order_))); ax.set_xticklabels([CONFIGS[c]["label"].replace(", ", ",\n").replace(" + ", "\n+ ") for c in order_], fontsize=7)
ax.set_ylabel("test MAE (veh/15 min)"); ax.set_ylim(min(m_.test_MAE_mean) * 0.9, max(m_.test_MAE_mean) * 1.05)
ax.set_title("Test MAE of main models and ablations (bars: mean ± std; dots: individual seeds)", fontsize=9)
savefig("q6a_ablation_bars.png"); plt.show()

# %%
# predicted vs actual traffic flow for three sensors (test period), 15-min and 60-min horizons (seed 42 models)
PRED = {c: np.load(os.path.join(RUN_DIR, f"pred_{c}_s{SEED}.npy")) for c in MAIN}
test_t = IDX["test"]
days = (ts15[test_t + 1] >= "2021-04-06") & (ts15[test_t + 1] < "2021-04-09")   # Tue 6 – Thu 8 Apr 2021
fig, ax = plt.subplots(len(PLOT_SENSORS), 2, figsize=(16, 3.2 * len(PLOT_SENSORS)), sharex="col")
for si, s in enumerate(PLOT_SENSORS):
    for hj, h in enumerate([0, CFG["t_out"] - 1]):
        a_ = ax[si, hj]
        tt = test_t[days] + 1 + h
        a_.plot(ts15[tt], np.where(OBS[tt, s], X15_raw[tt, s], np.nan), c="k", lw=1.6, label="actual")
        for c in MAIN:
            a_.plot(ts15[tt], PRED[c][days, h, si], c=colors[c], lw=1, label=CONFIGS[c]["label"])
        a_.set_title(f"sensor idx {s} (ID {sd_ids[s]}, {sd_meta.Fwy.values[s]}) — {15 * (h + 1)}-min-ahead", fontsize=9)
        a_.set_ylabel("veh / 15 min")
ax[0, 0].legend(fontsize=7, ncol=3)
fig.suptitle(f"Q6(a) predicted vs actual flow, test period 6–8 Apr 2021 (seed {SEED})")
savefig("q6a_pred_vs_actual.png"); plt.show()

# %% [markdown]
# ---
# # Question 5 / 6(b) — Learned graph structure analysis
# For each seed of the adaptive model we compare the learned top-20 graph with the road graph on **reliable sensors only**
# (both endpoints with ≥ 50 % observed 2020 data): (1) overlap with road edges and hop distance in the road graph,
# (2) geographic (haversine) distance, (3) correlation of 2020 flow series. Random pairs of reliable sensors are the null baseline.

# %%
from scipy.sparse.csgraph import shortest_path

def haversine_km(lat, lng):
    lat, lng = np.radians(lat), np.radians(lng)
    dlat = lat[:, None] - lat[None]; dlng = lng[:, None] - lng[None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin(dlng / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

GEO = haversine_km(sd_meta.Lat.values, sd_meta.Lng.values)
HOPS_UND = shortest_path(Abin.astype(float), unweighted=True, directed=False)
ROAD = Abin | Abin.T
REL = SRC_OK[:, None] & SRC_OK[None, :] & offdiag          # analysable pairs

def load_A(c, s, part=""):
    return np.load(os.path.join(RUN_DIR, f"A_{c}_s{s}{part}.npy"))

def edge_set(M, k_per_row):
    M = np.where(REL, M, 0.0)
    cols = np.argsort(-M, axis=1)[:, :k_per_row]
    rows_ = np.repeat(np.arange(len(M)), k_per_row)
    cols = cols.ravel(); keep_ = (M[rows_, cols] > 0) & SRC_OK[rows_]
    return rows_[keep_], cols[keep_]

def describe_edges(name, r_, c_):
    hop = HOPS_UND[r_, c_]
    return {"edges": name, "n": len(r_), "% road edge": 100 * ROAD[r_, c_].mean(),
            "% within 2 hops": 100 * (hop <= 2).mean(), "% unreachable": 100 * np.isinf(hop).mean(),
            "median hops": float(np.median(hop[np.isfinite(hop)])) if np.isfinite(hop).any() else np.nan,
            "median km": float(np.median(GEO[r_, c_])), "mean train corr": float(np.nanmean(CORR[r_, c_]))}

rs_ = np.random.RandomState(SEED)
rel_idx = np.where(SRC_OK)[0]
rr = rel_idx[rs_.randint(0, len(rel_idx), 20000)]; rc = rel_idx[rs_.randint(0, len(rel_idx), 20000)]
kk = rr != rc; rr, rc = rr[kk], rc[kk]
road_r, road_c = np.where(Abin & REL)

rows = [describe_edges("road graph edges", road_r, road_c)]
for s in CFG["seeds"]:
    A_ = load_A("adaptive", s)
    for k in (1, 5, CFG["topk"]):
        rows.append({**describe_edges(f"adaptive s{s}: top-{k}", *edge_set(A_, k)), "seed": s, "k": k})
    rows.append({**describe_edges(f"hybrid(adaptive part) s{s}: top-1", *edge_set(load_A("hybrid", s, "_adaptive_part"), 1)),
                 "seed": s, "k": 1, "hybrid": True})
rows.append(describe_edges("random reliable pairs", rr, rc))
GA = pd.DataFrame(rows)
GA.to_csv(os.path.join(OUT_DIR, "graph_analysis_per_seed.csv"), index=False)
num_cols = ["% road edge", "% within 2 hops", "% unreachable", "median hops", "median km", "mean train corr"]
summ = []
summ.append(GA.iloc[0][["edges"] + num_cols].to_dict())
for k in (1, 5, CFG["topk"]):
    sub = GA[(GA.k == k) & (GA.get("hybrid") != True) & GA.edges.str.startswith("adaptive")]
    d = {"edges": f"adaptive top-{k} (mean ± std, {len(sub)} seeds)"}
    for col in num_cols:
        d[col] = f"{sub[col].mean():.2f} ± {sub[col].std(ddof=1) if len(sub) > 1 else 0:.2f}"
    summ.append(d)
sub = GA[GA.get("hybrid") == True]
summ.append({"edges": f"hybrid adaptive part top-1 ({len(sub)} seeds)",
             **{col: f"{sub[col].mean():.2f} ± {sub[col].std(ddof=1) if len(sub) > 1 else 0:.2f}" for col in num_cols}})
summ.append(GA.iloc[-1][["edges"] + num_cols].to_dict())
GRAPH_SUMMARY = pd.DataFrame(summ)
GRAPH_SUMMARY.to_csv(os.path.join(OUT_DIR, "graph_analysis.csv"), index=False)
print(GRAPH_SUMMARY.to_string(index=False))

# stability of the learned graph across seeds (Jaccard overlap of each node's top-5 neighbour set)
def topk_sets(M, k=5):
    M = np.where(REL, M, 0.0)
    return [set(np.argsort(-M[i])[:k]) for i in rel_idx]
if len(CFG["seeds"]) > 1:
    sets = {s: topk_sets(load_A("adaptive", s)) for s in CFG["seeds"]}
    jac = [np.mean([len(a & b) / len(a | b) for a, b in zip(sets[s1], sets[s2])])
           for i, s1 in enumerate(CFG["seeds"]) for s2 in CFG["seeds"][i + 1:]]
    chance = 5 / (len(rel_idx) - 1)
    JACCARD = {"mean_top5_jaccard_across_seeds": float(np.mean(jac)), "chance_level_approx": float(chance / (2 - chance))}
    print("\nlearned-graph stability:", {k: round(v, 4) for k, v in JACCARD.items()})
else:
    JACCARD = {}

# strongest individual learned connections (seed 42, reliable pairs only)
A42 = load_A("adaptive", SEED)
Moff = np.where(REL, A42, 0.0)
flat = np.argsort(-Moff.ravel())[:15]
top_edges = pd.DataFrame({"i": flat // N, "j": flat % N})
top_edges["w"] = Moff.ravel()[flat].round(3)
top_edges["ID_i"] = sd_ids[top_edges.i]; top_edges["ID_j"] = sd_ids[top_edges.j]
top_edges["Fwy_i"] = sd_meta.Fwy.values[top_edges.i]; top_edges["Fwy_j"] = sd_meta.Fwy.values[top_edges.j]
top_edges["road_edge"] = ROAD[top_edges.i, top_edges.j]
top_edges["hops"] = HOPS_UND[top_edges.i, top_edges.j]
top_edges["geo_km"] = GEO[top_edges.i, top_edges.j].round(2)
top_edges["train_corr"] = CORR[top_edges.i, top_edges.j].round(3)
top_edges.to_csv(os.path.join(OUT_DIR, "top_learned_edges.csv"), index=False)
print(top_edges.to_string(index=False))
nr = top_edges[~top_edges.road_edge]
print(f"top-15: {int(top_edges.road_edge.sum())} road edges; {len(nr)} others, hops {sorted(nr.hops.tolist())}")

in_strength = A42.sum(0)
r_, c_ = edge_set(A42, CFG["topk"])
HUB = {"max_in_strength": float(in_strength.max()), "nodes_receiving_weight": int((in_strength > 1e-6).sum()),
       "top10_share": float(np.sort(in_strength)[-10:].sum() / N), "mean_self_weight": float(np.diag(A42).mean()),
       "spearman_weight_corr": float(pd.Series(A42[r_, c_]).corr(pd.Series(CORR[r_, c_]), method="spearman"))}
print("hub / weight statistics (seed 42):", {k: round(v, 3) for k, v in HUB.items()})

# %%
with torch.no_grad():
    A_HYB42 = load_A("hybrid", SEED)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a_, M, t_ in [(ax[0], A_FIXED, "fixed $\\hat A = D^{-1}\\tilde A$"),
                  (ax[1], A42, f"learned $A_{{adaptive}}$ (top-20, softmax), seed {SEED}"),
                  (ax[2], A_HYB42, f"hybrid ($\\alpha$={RES[('hybrid', SEED)]['alpha_final']:.2f}), seed {SEED}")]:
    im = a_.imshow(np.where(M > 1e-6, M, np.nan), cmap="viridis", interpolation="nearest",
                   norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1))
    a_.set_title(t_, fontsize=10); a_.grid(False)
fig.colorbar(im, ax=ax, fraction=0.02, label="weight (log scale, zeros blank)")
savefig("q5_adjacency_fixed_adaptive_hybrid.png"); plt.show()

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
r1, c1 = edge_set(A42, 1); r5, c5 = edge_set(A42, 5)
bins = np.linspace(0, np.percentile(GEO, 99), 60)
for (r_, c_), lab, col in [((road_r, road_c), "road-graph edges", "tab:blue"), ((r1, c1), "learned top-1", "tab:red"),
                           ((r5, c5), "learned top-5", "tab:orange"), ((rr, rc), "random pairs", "tab:gray")]:
    ax[0].hist(GEO[r_, c_], bins=bins, density=True, histtype="step", lw=1.6, label=lab, color=col)
    ax[1].hist(CORR[r_, c_], bins=np.linspace(-0.2, 1, 60), density=True, histtype="step", lw=1.6, label=lab, color=col)
ax[0].set_xlabel("geographic distance (km)"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
ax[0].set_title(f"Q6(b) distance between connected sensors (seed {SEED})")
ax[1].set_xlabel("Pearson corr. of 2020 flow"); ax[1].set_title("flow correlation between connected sensors")
hop_cats = ["1", "2", "3", "4-5", "6+", "unreach."]
def hop_hist(r_, c_):
    h = HOPS_UND[r_, c_]
    return np.array([(h == 1).mean(), (h == 2).mean(), (h == 3).mean(), ((h >= 4) & (h <= 5)).mean(),
                     ((h >= 6) & np.isfinite(h)).mean(), np.isinf(h).mean()]) * 100
xb = np.arange(len(hop_cats)); wdt = 0.27
for o, (r_, c_), lab, col in [(-wdt, (r1, c1), "learned top-1", "tab:red"), (0, (r5, c5), "learned top-5", "tab:orange"),
                              (wdt, (rr, rc), "random pairs", "tab:gray")]:
    ax[2].bar(xb + o, hop_hist(r_, c_), wdt, label=lab, color=col)
ax[2].set_xticks(xb); ax[2].set_xticklabels(hop_cats); ax[2].set_xlabel("hops in road graph (undirected)")
ax[2].set_ylabel("% of edges"); ax[2].legend(fontsize=8); ax[2].set_title("road-graph hop distance of learned edges")
savefig("q6b_learned_vs_physical.png"); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(15, 7))
for a_, (r_, c_), t_ in [(ax[0], (road_r, road_c), "road-distance graph (directed edges, reliable sensors)"),
                         (ax[1], (r1, c1), "learned graph: strongest neighbour of each node")]:
    a_.scatter(sd_meta.Lng, sd_meta.Lat, s=4, c="k", zorder=3)
    for i_, j_, p_ in zip(r_, c_, ROAD[r_, c_]):
        a_.plot([sd_meta.Lng.values[i_], sd_meta.Lng.values[j_]], [sd_meta.Lat.values[i_], sd_meta.Lat.values[j_]],
                c=("tab:blue" if p_ else "tab:red"), lw=0.5, alpha=0.5)
    a_.set_title(t_ + ("" if a_ is ax[0] else "  (blue = also a road edge, red = not)"), fontsize=9)
    a_.set_xlabel("longitude"); a_.set_ylabel("latitude"); a_.set_aspect("equal")
savefig("q6b_map_learned_edges.png"); plt.show()

# %% [markdown]
# ---
# # Save everything (Q7)

# %%
def to_py(o):
    if isinstance(o, dict): return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.bool_): return bool(o)
    return o

summary = {
    "config": CFG, "hardware": HW, "smoke_test": SMOKE,
    "timing": {"load_h5_s": LOAD_TIME, "total_training_wall_min": TOTAL_TRAIN_WALL / 60, "n_workers": n_workers,
               "peak_rss_notebook_gb_after_prep": MEM_PREP_GB, "peak_rss_notebook_gb_end": peak_rss_gb()},
    "data": {"n_sensors": int(N), "t15": int(len(ts15)), "start": str(ts15[0]), "end": str(ts15[-1]),
             "missing_5min_frac": MISS_5MIN, "partial_15min_frac": PARTIAL_15, "missing_15min_frac_after_faults": MISS_15,
             "zero_stats": ZERO_STATS, "inputs_forward_filled": int(n_ffill), "inputs_profile_filled": int(n_prof),
             "train_mean": MU, "train_std": SD, "windows": {k: int(len(v)) for k, v in IDX.items()},
             "reliable_sources": int(SRC_OK.sum()), "unreliable_sources": int((~SRC_OK).sum()),
             "sensors_no_2020_obs": int((train_obs_frac == 0).sum()),
             "road_edges_removed_unreliable_source": int(Abin[:, ~SRC_OK].sum()),
             "plot_sensors": PLOT_SENSORS, "selected_sensor_q3c": SEL,
             "q3c": {"one_hop": [int(sd_ids[j]) for j in one_hop], "two_hop_only": [int(sd_ids[j]) for j in two_only]}},
    "graph_stats": graph_stats, "params": PARAMS,
    "summary": SUMMARY.to_dict(orient="records"),
    "paired": PAIRED.to_dict(orient="records"),
    "naive_reference_test": {k: {"MAE": v[0], "RMSE": v[1]} for k, v in NAIVE.items()},
    "graph_analysis": GRAPH_SUMMARY.to_dict(orient="records"), "graph_stability": JACCARD, "hub_stats": HUB,
    "q1c_table": q1c_rows, "q2": {"a_ls": float(a_ls), "b_ls": float(b_ls)},
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(to_py(summary), f, indent=2, default=str)
shutil.rmtree(os.path.join(RUN_DIR, "claims"), ignore_errors=True)
shutil.copy(LIB_PATH, os.path.join(OUT_DIR, "stgnn_lib.py"))
print("saved results.json; files in", OUT_DIR, ":", sorted(os.listdir(OUT_DIR)))
print("figures:", sorted(os.listdir(FIG_DIR)))
print(f"peak RAM of notebook process: {peak_rss_gb():.2f} GB")
