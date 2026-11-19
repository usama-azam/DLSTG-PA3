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
# | Q3 | LargeST San Diego extraction (716 sensors, 2020–2021), 5 → 15 min resampling, missing values, road-graph analysis, 1/2-hop neighbourhoods |
# | Q4 | Forecasting windows (8 → 4), chronological splits, temporal-conv + GCN + head model, temporal-only baseline |
# | Q5 | Fixed vs learnable (adaptive) vs hybrid adjacency, trained with an identical protocol |
# | Q6 | Comparison table, predicted-vs-actual plots, learned-graph analysis |
# | Q7 | Hyper-parameters, seed, hardware and timings are printed and saved to `outputs/` |
#
# The notebook is designed to run on Kaggle with the dataset `liuxu77/largest` attached
# (GPU optional). If the dataset is not found, it falls back to a small **synthetic smoke test**
# so that the code path can be checked on any machine.

# %%
import os, sys, json, time, math, random, platform, warnings
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

# ----------------------------------------------------------------------------- paths
CANDIDATE_DIRS = ["/kaggle/input/datasets/liuxu77/largest", "/kaggle/input/largest",
                  os.environ.get("LARGEST_DIR", "data/largest")]
DATA_DIR = next((d for d in CANDIDATE_DIRS if os.path.exists(os.path.join(d, "ca_meta.csv"))), None)
SMOKE = DATA_DIR is None
OUT_DIR = "/kaggle/working/outputs" if os.path.exists("/kaggle/working") else "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------------- config
CFG = dict(
    seed=42,
    district=11,                 # San Diego = Caltrans District 11
    years=[2020, 2021],
    agg="sum",                   # PeMS 5-min flow is a vehicle COUNT per interval -> sum to 15 min
    t_in=8, t_out=4,             # 2 h history -> 1 h horizon at 15-min resolution
    split=dict(train=("2020-01-01", "2020-12-31 23:45"),
               val=("2021-01-01", "2021-03-31 23:45"),
               test=("2021-04-01", "2021-12-31 23:45")),
    short_gap_steps=4,           # gaps <= 1 h: linear interpolation; longer: train time-of-week profile
    hidden=32,                   # d (assignment: 16-32)
    gcn_layers=2,
    emb_dim=10,                  # d_e of the node embeddings E1, E2
    topk=20,                     # sparsify adaptive adjacency to top-k neighbours per node
    alpha_init=0.5,              # hybrid: alpha = sigmoid(a), learnable, initialised at 0.5
    batch_size=64,
    lr=1e-3,
    weight_decay=0.0,
    max_epochs=50,
    patience=8,                  # early stopping on validation MAE
    grad_clip=5.0,
    loss="masked MAE",
)
if SMOKE:
    CFG.update(max_epochs=2, patience=2)
print("DATA_DIR:", DATA_DIR, "| SMOKE TEST:", SMOKE, "| OUT_DIR:", OUT_DIR)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path)
    print("saved", path)

# %% [markdown]
# ## Hardware and software (Q7)

# %%
def hardware_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "cpu": platform.processor() or "unknown", "cpu_count": os.cpu_count(),
            "device": str(DEVICE)}
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
print(f"5-min tensor {X5.shape}, loaded in {time.time() - t0:.1f}s")

# %%
# ---- 5 -> 15 min aggregation (sum of counts; rescaled mean when partially missing)
T15 = X5.shape[0] // 3
X5r = X5[:T15 * 3].reshape(T15, 3, N)
n_obs = np.isfinite(X5r).sum(1)
with np.errstate(invalid="ignore"):
    X15_raw = np.where(n_obs > 0, np.nanmean(X5r, axis=1) * 3.0, np.nan).astype(np.float32)
ts15 = ts5[::3][:T15]
OBS = np.isfinite(X15_raw)                     # True where a real 15-min measurement exists
print("15-min tensor:", X15_raw.shape, ts15[0], "->", ts15[-1])
print(f"missing 5-min readings : {np.isnan(X5).mean():.4%}")
print(f"15-min with partial data (1-2 of 3 present): {((n_obs > 0) & (n_obs < 3)).mean():.4%}")
print(f"15-min fully missing   : {(~OBS).mean():.4%}")
miss_sensor = 1 - OBS.mean(0)
print(f"per-sensor missing rate: median {np.median(miss_sensor):.3%}, max {miss_sensor.max():.3%}, "
      f"sensors >10% missing: {(miss_sensor > 0.10).sum()}, sensors 100% missing: {(miss_sensor == 1).sum()}")
del X5r

# %% [markdown]
# **Missing-value handling.** (1) A boolean mask `OBS` records which 15-min values were really observed;
# all losses and metrics are computed **only on observed targets** (masked MAE/RMSE), so imputed values never
# count as ground truth. (2) For the model *inputs*, short gaps (≤ 1 h = 4 steps) are linearly interpolated in
# time; longer gaps are filled with the sensor's **training-period (2020) time-of-week mean profile**; any residue
# falls back to the sensor's training mean. Only 2020 statistics are used, so no validation/test information leaks.

# %%
def split_bounds(name):
    s, e = CFG["split"][name]
    return int(np.searchsorted(ts15, pd.Timestamp(s))), int(np.searchsorted(ts15, pd.Timestamp(e), side="right")) - 1

TR, VA, TE = split_bounds("train"), split_bounds("val"), split_bounds("test")
print("index bounds  train", TR, " val", VA, " test", TE)

def fill_missing(X, obs, train_end, max_gap):
    X = X.copy()
    T, n = X.shape
    tow = (ts15.dayofweek * 96 + ts15.hour * 4 + ts15.minute // 15).values   # 0..671
    # training time-of-week profile (observed values only)
    prof = np.full((672, n), np.nan, dtype=np.float32)
    tr = slice(0, train_end + 1)
    df_tr = pd.DataFrame(np.where(obs[tr], X[tr], np.nan)); df_tr["tow"] = tow[tr]
    g = df_tr.groupby("tow").mean()
    prof[g.index.values] = g.values
    sensor_mean = np.nanmean(np.where(obs[tr], X[tr], np.nan), axis=0)
    sensor_mean = np.where(np.isfinite(sensor_mean), sensor_mean, np.nanmean(sensor_mean))
    prof = np.where(np.isfinite(prof), prof, sensor_mean[None])
    n_short = n_long = 0
    for j in range(n):
        m = ~obs[:, j]
        if not m.any():
            continue
        # run-length of missing runs
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            L = e - s
            if L <= max_gap and s > 0 and e < T:      # X[s-1] and X[e] are observed (run boundaries)
                frac = np.arange(1, L + 1) / (L + 1)
                X[s:e, j] = X[s - 1, j] + (X[e, j] - X[s - 1, j]) * frac; n_short += L
            else:
                X[s:e, j] = prof[tow[s:e], j]; n_long += L
    return X, prof, n_short, n_long

t0 = time.time()
X15, TOW_PROFILE, n_short, n_long = fill_missing(X15_raw, OBS, TR[1], CFG["short_gap_steps"])
print(f"filled {n_short} values by interpolation and {n_long} by train time-of-week profile "
      f"({time.time() - t0:.1f}s); remaining NaN: {np.isnan(X15).sum()}")

# %%
# ---- Plot: three sensors over one week (+ missing-data overview)
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

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].hist(miss_sensor * 100, bins=50, color="tab:gray"); ax[0].set_xlabel("missing rate per sensor (%)")
ax[0].set_ylabel("# sensors"); ax[0].set_yscale("log")
daily = pd.Series((~OBS).mean(1), index=ts15).resample("D").mean() * 100
ax[1].plot(daily.index, daily.values, lw=1); ax[1].set_ylabel("missing (%) per day")
for a_ in ax: a_.set_title("Q3(a) missing 15-min values", fontsize=9)
savefig("q3a_missing.png"); plt.show()

# %% [markdown]
# ### Q3(b) Road-distance adjacency for the 716 sensors
# `ca_rn_adj.npy` (8 600 × 8 600) is indexed by the row order of `ca_meta.csv`; the h5 columns use the same
# order (asserted below), so `A_SD = A[sd_rows][:, sd_rows]` matches the column order of the traffic tensor.

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
graph_stats = {
    "nodes": int(N),
    "nonzero_entries_incl_self_loops": int(nz.sum()),
    "self_loops": int(np.diag(nz).sum()),
    "directed_edges_excl_self": int((nz & offdiag).sum()),
    "undirected_pairs": int(np.triu((nz | nz.T) & offdiag).sum()),
    "symmetric_exact": bool(np.array_equal(A_SD, A_SD.T)),
    "symmetric_pattern": bool(np.array_equal(nz, nz.T)),
    "asymmetric_pairs_pattern": int(np.triu((nz != nz.T)).sum()),
    "max_abs_A_minus_AT": float(np.abs(A_SD - A_SD.T).max()),
    "weight_min_nonzero": float(A_SD[nz & offdiag].min()) if (nz & offdiag).any() else None,
    "weight_max_offdiag": float(A_SD[offdiag].max()),
    "density_offdiag": float((nz & offdiag).sum() / (N * (N - 1))),
}
OUT_DEG = (nz & offdiag).sum(1); IN_DEG = (nz & offdiag).sum(0)
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

w = A_SD[nz & offdiag]
plt.figure(figsize=(5, 3)); plt.hist(w, bins=50); plt.xlabel("edge weight $A_{ij}$"); plt.ylabel("# edges")
plt.title("Q3(b) distribution of nonzero off-diagonal weights", fontsize=9)
savefig("q3b_weights.png"); plt.show()

# %% [markdown]
# ### Q3(c) One-hop and two-hop neighbourhoods of a selected sensor

# %%
Abin = (nz & offdiag)
two_hop_reach = ((Abin.astype(np.int32) @ Abin.astype(np.int32)) > 0) | Abin
np.fill_diagonal(two_hop_reach, False)
cand = np.where(OUT_DEG == int(np.median(OUT_DEG)))[0]
SEL = int(cand[0]) if len(cand) else int(np.argsort(OUT_DEG)[N // 2])
one_hop = np.where(Abin[SEL])[0]
two_hop = np.where(two_hop_reach[SEL])[0]
two_only = np.setdiff1d(two_hop, one_hop)
r = sd_meta.iloc[SEL]
print(f"selected sensor idx {SEL} (ID {r.ID}, {r.Fwy}, lat {r.Lat:.4f}, lng {r.Lng:.4f})")
print(f"1-hop neighbours ({len(one_hop)}):", [int(sd_ids[j]) for j in one_hop])
print(f"  weights:", np.round(A_SD[SEL, one_hop], 3).tolist())
print(f"2-hop neighbourhood (<=2 edges) size {len(two_hop)}; reachable only in exactly 2 hops: {len(two_only)}")
print("  2-hop-only IDs:", [int(sd_ids[j]) for j in two_only])
in_nb = np.where(Abin[:, SEL])[0]
print(f"in-neighbours (j with A_j,sel != 0): {len(in_nb)}")
nbr_table = pd.DataFrame({"idx": np.r_[one_hop, two_only], "hop": [1] * len(one_hop) + [2] * len(two_only)})
nbr_table["ID"] = sd_ids[nbr_table.idx]; nbr_table["Fwy"] = sd_meta.Fwy.values[nbr_table.idx]
nbr_table["A_sel_j"] = A_SD[SEL, nbr_table.idx]
corr_sel = [np.corrcoef(X15[TR[0]:TR[1] + 1, SEL], X15[TR[0]:TR[1] + 1, j])[0, 1] for j in nbr_table.idx]
nbr_table["train_corr"] = np.round(corr_sel, 3)
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

# tensors on the device: normalised filled inputs, raw targets and mask
DATA = torch.tensor((X15 - MU) / SD, dtype=torch.float32, device=DEVICE)          # [T, N]
RAW = torch.tensor(np.nan_to_num(X15_raw), dtype=torch.float32, device=DEVICE)    # [T, N]
MASK = torch.tensor(OBS, dtype=torch.bool, device=DEVICE)                         # [T, N]
OFF_IN = torch.arange(-CFG["t_in"] + 1, 1, device=DEVICE)
OFF_OUT = torch.arange(1, CFG["t_out"] + 1, device=DEVICE)

def get_batch(t_idx):
    t_idx = torch.as_tensor(t_idx, device=DEVICE)
    x = DATA[t_idx[:, None] + OFF_IN]                 # [B, 8, N]
    yi = t_idx[:, None] + OFF_OUT
    return x, DATA[yi], RAW[yi], MASK[yi]            # y_norm, y_raw, mask: [B, 4, N]

# %% [markdown]
# ### Q4(b) Model: temporal conv encoder → graph convolution(s) → forecast head
# * **Temporal encoder** (shared by all sensors): `Conv1d(1→d, k=3)` → ReLU → `Conv1d(d→d, k=3, dilation=2)` → ReLU
#   (receptive field = all 8 steps) → flatten → `Linear(2d→d)` → ReLU, giving $Z\in\mathbb R^{B\times N\times d}$.
# * **Graph convolution**: $H=\mathrm{ReLU}(\hat A Z W_g)$, implemented as `einsum('nm,bmd->bnd')` so the graph
#   product acts over the node dimension independently for each batch item. We stack `gcn_layers` such layers, each
#   with a residual connection $H^{(l+1)}=H^{(l)}+\mathrm{ReLU}(\hat A H^{(l)} W^{(l)})$ so that the graph model
#   strictly contains the temporal-only model (fair comparison) and adaptive graphs cannot wash out a node's own signal.
# * **Head**: per-node MLP `Linear(d→d)` → ReLU → `Linear(d→4)`.
# * **Temporal-only baseline**: identical encoder and head, graph layers removed.

# %%
class TemporalEncoder(nn.Module):
    def __init__(self, d, t_in):
        super().__init__()
        self.c1 = nn.Conv1d(1, d, 3)
        self.c2 = nn.Conv1d(d, d, 3, dilation=2)
        L = t_in - 2 - 4
        self.proj = nn.Linear(d * L, d)

    def forward(self, x):                     # x: [B, T, N]
        B, T, N_ = x.shape
        h = x.permute(0, 2, 1).reshape(B * N_, 1, T)
        h = F.relu(self.c2(F.relu(self.c1(h))))
        return F.relu(self.proj(h.flatten(1))).view(B, N_, -1)   # [B, N, d]


class GraphConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d)

    def forward(self, Z, A):                  # Z: [B, N, d], A: [N, N] (row i aggregates from columns j)
        return F.relu(torch.einsum("nm,bmd->bnd", A, self.W(Z)))


def normalize_fixed(A):
    """Directed random-walk normalisation  Â = D^{-1} Ã,  Ã = A with unit self-loops, D_ii = sum_j Ã_ij."""
    At = A.copy(); np.fill_diagonal(At, 1.0)
    return (At / At.sum(1, keepdims=True)).astype(np.float32)


class FixedAdj(nn.Module):
    def __init__(self, A_hat):
        super().__init__()
        self.register_buffer("A", torch.tensor(A_hat))

    def forward(self):
        return self.A


class AdaptiveAdj(nn.Module):
    """A_adp = Softmax_row( ReLU(E1 E2^T) ), optionally sparsified to the top-k logits of each row
    BEFORE the softmax (so each row is a distribution over its k kept neighbours)."""
    def __init__(self, n, de, topk=None):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(n, de))
        self.E2 = nn.Parameter(torch.randn(n, de))
        self.topk = topk

    def forward(self):
        logits = F.relu(self.E1 @ self.E2.T)
        if self.topk and self.topk < logits.shape[1]:
            vals, idx = logits.topk(self.topk, dim=1)
            logits = torch.full_like(logits, float("-inf")).scatter(1, idx, vals)
        return torch.softmax(logits, dim=1)


class HybridAdj(nn.Module):
    """A_hyb = alpha * Â_fixed + (1 - alpha) * A_adp, both row-stochastic; alpha = sigmoid(a) is learned."""
    def __init__(self, A_hat, n, de, topk, alpha_init=0.5):
        super().__init__()
        self.fixed = FixedAdj(A_hat)
        self.adaptive = AdaptiveAdj(n, de, topk)
        self.a = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    @property
    def alpha(self):
        return torch.sigmoid(self.a)

    def forward(self):
        return self.alpha * self.fixed() + (1 - self.alpha) * self.adaptive()


class STGNN(nn.Module):
    def __init__(self, n, d, t_in, t_out, adj=None, gcn_layers=1):
        super().__init__()
        self.encoder = TemporalEncoder(d, t_in)
        self.adj = adj
        self.gcns = nn.ModuleList([GraphConv(d) for _ in range(gcn_layers if adj is not None else 0)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, t_out))

    def forward(self, x):                      # x: [B, T_in, N] -> [B, T_out, N]
        H = self.encoder(x)
        if self.adj is not None:
            A = self.adj()
            for g in self.gcns:
                H = H + g(H, A)
        return self.head(H).permute(0, 2, 1)


A_HAT = normalize_fixed(A_SD)

def build_model(kind):
    set_seed(CFG["seed"])                       # identical initialisation of shared parts
    d, L = CFG["hidden"], CFG["gcn_layers"]
    adj = {"none": None,
           "fixed": lambda: FixedAdj(A_HAT),
           "adaptive": lambda: AdaptiveAdj(N, CFG["emb_dim"], CFG["topk"]),
           "hybrid": lambda: HybridAdj(A_HAT, N, CFG["emb_dim"], CFG["topk"], CFG["alpha_init"])}[kind]
    return STGNN(N, d, CFG["t_in"], CFG["t_out"], adj() if adj else None, L).to(DEVICE)

for k in ["none", "fixed", "adaptive", "hybrid"]:
    m = build_model(k)
    print(f"{k:>8}: {sum(p.numel() for p in m.parameters() if p.requires_grad):,} trainable parameters")

# %% [markdown]
# ### Training / evaluation protocol (identical for every model)
# Adam (lr 1e-3), batch 64, masked-MAE loss in normalised units, gradient clipping 5, up to 50 epochs with early
# stopping on validation MAE (patience 8) and restoration of the best weights. Metrics are computed in **original
# units (vehicles / 15 min)** on observed targets only. Training time = wall-clock time of the full training loop
# (including per-epoch validation), measured with CUDA synchronisation.

# %%
def masked_mae_loss(pred, y, m):
    m = m.float()
    return (torch.abs(pred - y) * m).sum() / m.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, t_idx, bs=256, keep_sensors=None):
    model.eval()
    abs_e = torch.zeros(CFG["t_out"], device=DEVICE); sq_e = torch.zeros_like(abs_e); cnt = torch.zeros_like(abs_e)
    kept = []
    for i in range(0, len(t_idx), bs):
        x, _, y_raw, m = get_batch(t_idx[i:i + bs])
        p = model(x) * SD + MU
        mf = m.float()
        abs_e += (torch.abs(p - y_raw) * mf).sum((0, 2)); sq_e += (((p - y_raw) ** 2) * mf).sum((0, 2))
        cnt += mf.sum((0, 2))
        if keep_sensors is not None:
            kept.append(p[:, :, keep_sensors].cpu())
    mae_h, rmse_h = (abs_e / cnt).cpu().numpy(), torch.sqrt(sq_e / cnt).cpu().numpy()
    out = {"MAE": float(abs_e.sum() / cnt.sum()), "RMSE": float(torch.sqrt(sq_e.sum() / cnt.sum())),
           "MAE_h": mae_h.tolist(), "RMSE_h": rmse_h.tolist()}
    return (out, torch.cat(kept).numpy()) if keep_sensors is not None else out


def train(kind):
    model = build_model(kind)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    gen = torch.Generator().manual_seed(CFG["seed"])
    tr_idx = IDX["train"]
    best, best_state, bad, hist = float("inf"), None, 0, []
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_start = time.time()
    for ep in range(1, CFG["max_epochs"] + 1):
        model.train(); te = time.time(); run, nb = 0.0, 0
        perm = tr_idx[torch.randperm(len(tr_idx), generator=gen).numpy()]
        for i in range(0, len(perm), CFG["batch_size"]):
            x, y, _, m = get_batch(perm[i:i + CFG["batch_size"]])
            loss = masked_mae_loss(model(x), y, m)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"]); opt.step()
            run += loss.item(); nb += 1
        val = evaluate(model, IDX["val"])
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        rec = {"epoch": ep, "train_mae_norm": run / nb, "train_mae": run / nb * SD,
               "val_mae": val["MAE"], "val_rmse": val["RMSE"], "epoch_time_s": time.time() - te}
        if hasattr(model.adj, "alpha"): rec["alpha"] = float(model.adj.alpha)
        hist.append(rec)
        print(f"[{kind:>8}] ep {ep:2d} train MAE {rec['train_mae']:.3f} | val MAE {val['MAE']:.3f} "
              f"RMSE {val['RMSE']:.3f} | {rec['epoch_time_s']:.1f}s" + (f" | alpha {rec['alpha']:.3f}" if "alpha" in rec else ""))
        if val["MAE"] < best - 1e-4:
            best, bad = val["MAE"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_ep = ep
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    train_time = time.time() - t_start
    model.load_state_dict(best_state)
    return model, {"train_time_s": train_time, "epochs_run": ep, "best_epoch": best_ep,
                   "sec_per_epoch": train_time / ep, "history": hist}

# %% [markdown]
# ### Train the four models (Q4(b), Q4(c), Q5(a)–(c))

# %%
MODELS, RESULTS, PRED_KEEP = {}, {}, {}
KEEP = PLOT_SENSORS
LABELS = {"none": ("Temporal-only", "None"), "fixed": ("STGNN", "Fixed"),
          "adaptive": ("STGNN", "Adaptive"), "hybrid": ("STGNN", "Hybrid")}
for kind in ["none", "fixed", "adaptive", "hybrid"]:
    model, info = train(kind)
    val = evaluate(model, IDX["val"])
    test, kept = evaluate(model, IDX["test"], keep_sensors=KEEP)
    info.update(val=val, test=test)
    if hasattr(model.adj, "alpha"): info["alpha_final"] = float(model.adj.alpha)
    MODELS[kind], RESULTS[kind], PRED_KEEP[kind] = model, info, kept
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"model_{kind}.pt"))
    print(f"==> {kind}: val MAE {val['MAE']:.3f} RMSE {val['RMSE']:.3f} | test MAE {test['MAE']:.3f} "
          f"RMSE {test['RMSE']:.3f} | train {info['train_time_s']:.1f}s ({info['epochs_run']} ep, best {info['best_epoch']})")

# %% [markdown]
# ---
# # Question 6(a) — Forecasting comparison

# %%
rows = []
for kind, r in RESULTS.items():
    mdl, adj = LABELS[kind]
    rows.append({"Model": mdl, "Adjacency": adj,
                 "Val MAE": r["val"]["MAE"], "Val RMSE": r["val"]["RMSE"],
                 "Test MAE": r["test"]["MAE"], "Test RMSE": r["test"]["RMSE"],
                 **{f"Test MAE @{15 * (h + 1)}m": r["test"]["MAE_h"][h] for h in range(CFG["t_out"])},
                 **{f"Test RMSE @{15 * (h + 1)}m": r["test"]["RMSE_h"][h] for h in range(CFG["t_out"])},
                 "Train time (s)": r["train_time_s"], "Epochs": r["epochs_run"], "Best epoch": r["best_epoch"],
                 "s/epoch": r["sec_per_epoch"]})
TABLE = pd.DataFrame(rows)
TABLE.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
pd.set_option("display.width", 200)
print(TABLE[["Model", "Adjacency", "Val MAE", "Val RMSE", "Test MAE", "Test RMSE", "Train time (s)", "Epochs", "Best epoch"]]
      .round(3).to_string(index=False))
print()
print(TABLE[["Adjacency"] + [c for c in TABLE.columns if "@" in c]].round(3).to_string(index=False))

# naive reference points (not required; helps interpret the numbers)
def naive_metrics(kind):
    t_idx = IDX["test"]; abs_e = sq_e = cnt = 0.0
    tow = (ts15.dayofweek * 96 + ts15.hour * 4 + ts15.minute // 15).values
    for i in range(0, len(t_idx), 1024):
        ti = t_idx[i:i + 1024]
        yi = ti[:, None] + np.arange(1, 5)
        y, m = X15_raw[yi], OBS[yi]
        p = np.repeat(X15[ti][:, None], 4, 1) if kind == "last" else TOW_PROFILE[tow[yi]]
        e = np.where(m, p - np.nan_to_num(y), 0)
        abs_e += np.abs(e).sum(); sq_e += (e ** 2).sum(); cnt += m.sum()
    return abs_e / cnt, math.sqrt(sq_e / cnt)

NAIVE = {k: naive_metrics(k) for k in ["last", "tow"]}
print("\nreference (test): last-value MAE/RMSE = %.3f / %.3f ; train time-of-week profile MAE/RMSE = %.3f / %.3f"
      % (*NAIVE["last"], *NAIVE["tow"]))

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
colors = {"none": "tab:gray", "fixed": "tab:blue", "adaptive": "tab:orange", "hybrid": "tab:green"}
for kind, r in RESULTS.items():
    h = pd.DataFrame(r["history"])
    ax[0].plot(h.epoch, h.train_mae, c=colors[kind], ls="--", lw=1)
    ax[0].plot(h.epoch, h.val_mae, c=colors[kind], label=LABELS[kind][1] if kind != "none" else "Temporal-only")
    ax[1].plot([15, 30, 45, 60], r["test"]["MAE_h"], "o-", c=colors[kind])
    ax[2].plot([15, 30, 45, 60], r["test"]["RMSE_h"], "o-", c=colors[kind])
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MAE (veh/15 min)"); ax[0].set_title("train (dashed) / val (solid) MAE")
ax[0].legend()
for a_, t_ in zip(ax[1:], ["test MAE per horizon", "test RMSE per horizon"]):
    a_.set_xlabel("horizon (min)"); a_.set_title(t_); a_.set_xticks([15, 30, 45, 60])
savefig("q6a_curves_horizons.png"); plt.show()

# %%
# predicted vs actual traffic flow for three sensors (test period), 15-min and 60-min horizons
test_t = IDX["test"]
days = (ts15[test_t + 1] >= "2021-04-06") & (ts15[test_t + 1] < "2021-04-09")   # Tue 6 – Thu 8 Apr 2021
fig, ax = plt.subplots(len(KEEP), 2, figsize=(16, 3.2 * len(KEEP)), sharex="col")
for si, s in enumerate(KEEP):
    for hj, h in enumerate([0, 3]):
        a_ = ax[si, hj]
        tt = test_t[days] + 1 + h
        truth = np.where(OBS[tt, s], X15_raw[tt, s], np.nan)
        a_.plot(ts15[tt], truth, c="k", lw=1.6, label="actual")
        for kind in RESULTS:
            a_.plot(ts15[tt], PRED_KEEP[kind][days, h, si], c=colors[kind], lw=1,
                    label=f"{LABELS[kind][0]} ({LABELS[kind][1]})")
        a_.set_title(f"sensor idx {s} (ID {sd_ids[s]}) — {15 * (h + 1)}-min-ahead", fontsize=9)
        a_.set_ylabel("veh / 15 min")
ax[0, 0].legend(fontsize=7, ncol=3)
fig.suptitle("Q6(a) predicted vs actual flow, test period 6–8 Apr 2021")
savefig("q6a_pred_vs_actual.png"); plt.show()

# %% [markdown]
# ---
# # Question 5 / 6(b) — Learned graph structure analysis
# For the adaptive model we extract $A_{adp}$ (top-k=20 per row) and compare it with the road-distance graph using
# (1) overlap with physical edges and hop distance in the road graph, (2) geographic (haversine) distance between
# connected sensors, and (3) correlation of 2020 (training) flow series between connected sensors.

# %%
with torch.no_grad():
    A_ADP = MODELS["adaptive"].adj().cpu().numpy()
    A_HYB = MODELS["hybrid"].adj().cpu().numpy()
    A_HYB_ADP = MODELS["hybrid"].adj.adaptive().cpu().numpy()
np.save(os.path.join(OUT_DIR, "A_adaptive.npy"), A_ADP.astype(np.float32))
np.save(os.path.join(OUT_DIR, "A_hybrid.npy"), A_HYB.astype(np.float32))
np.save(os.path.join(OUT_DIR, "A_fixed_normalized.npy"), A_HAT)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a_, M, t_ in [(ax[0], A_HAT, "fixed $\\hat A = D^{-1}\\tilde A$"),
                  (ax[1], A_ADP, "learned $A_{adaptive}$ (top-20, softmax)"),
                  (ax[2], A_HYB, f"hybrid ($\\alpha$={RESULTS['hybrid']['alpha_final']:.2f})")]:
    im = a_.imshow(np.where(M > 1e-6, M, np.nan), cmap="viridis", interpolation="nearest",
                   norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1))
    a_.set_title(t_); a_.grid(False)
fig.colorbar(im, ax=ax, fraction=0.02, label="weight (log scale, zeros blank)")
savefig("q5_adjacency_fixed_adaptive_hybrid.png"); plt.show()

# %%
from scipy.sparse.csgraph import shortest_path

def haversine_km(lat, lng):
    lat, lng = np.radians(lat), np.radians(lng)
    dlat = lat[:, None] - lat[None]; dlng = lng[:, None] - lng[None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin(dlng / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

GEO = haversine_km(sd_meta.Lat.values, sd_meta.Lng.values)
HOPS_DIR = shortest_path(Abin.astype(float), unweighted=True, directed=True)
HOPS_UND = shortest_path(Abin.astype(float), unweighted=True, directed=False)
CORR = np.corrcoef(X15[TR[0]:TR[1] + 1].T)      # training-period correlations only

def edge_set(M, k_per_row=None, thresh=None):
    M = M.copy(); np.fill_diagonal(M, 0)
    if k_per_row:
        cols = np.argsort(-M, axis=1)[:, :k_per_row]
        rows_ = np.repeat(np.arange(len(M)), k_per_row)
        cols = cols.ravel(); keep = M[rows_, cols] > 0
        return rows_[keep], cols[keep]
    r_, c_ = np.where(M > thresh); return r_, c_

def describe_edges(name, r_, c_):
    hop_u = HOPS_UND[r_, c_]
    return {"edges": name, "n": len(r_),
            "% in road graph (i->j)": 100 * Abin[r_, c_].mean(),
            "% in road graph (either dir)": 100 * (Abin | Abin.T)[r_, c_].mean(),
            "% within 2 hops": 100 * (hop_u <= 2).mean(),
            "% unreachable": 100 * np.isinf(hop_u).mean(),
            "median hops (finite)": float(np.median(hop_u[np.isfinite(hop_u)])) if np.isfinite(hop_u).any() else np.nan,
            "median geo km": float(np.median(GEO[r_, c_])), "mean geo km": float(GEO[r_, c_].mean()),
            "mean train corr": float(np.nanmean(CORR[r_, c_]))}

rs = np.random.RandomState(CFG["seed"])
rr = rs.randint(0, N, 20000); rc = rs.randint(0, N, 20000); keep = rr != rc
analysis = [describe_edges("road graph edges", *np.where(Abin)),
            describe_edges("adaptive: top-1 per node", *edge_set(A_ADP, 1)),
            describe_edges("adaptive: top-5 per node", *edge_set(A_ADP, 5)),
            describe_edges("adaptive: all kept (top-20)", *edge_set(A_ADP, CFG["topk"])),
            describe_edges("hybrid-adaptive: top-1 per node", *edge_set(A_HYB_ADP, 1)),
            describe_edges("random pairs", rr[keep], rc[keep])]
GRAPH_ANALYSIS = pd.DataFrame(analysis)
GRAPH_ANALYSIS.to_csv(os.path.join(OUT_DIR, "graph_analysis.csv"), index=False)
print(GRAPH_ANALYSIS.round(2).to_string(index=False))

# strongest individual learned connections
Moff = A_ADP.copy(); np.fill_diagonal(Moff, 0)
flat = np.argsort(-Moff.ravel())[:15]
top_edges = pd.DataFrame({"i": flat // N, "j": flat % N})
top_edges["w"] = Moff.ravel()[flat]
top_edges["ID_i"] = sd_ids[top_edges.i]; top_edges["ID_j"] = sd_ids[top_edges.j]
top_edges["Fwy_i"] = sd_meta.Fwy.values[top_edges.i]; top_edges["Fwy_j"] = sd_meta.Fwy.values[top_edges.j]
top_edges["road_edge"] = (Abin | Abin.T)[top_edges.i, top_edges.j]
top_edges["hops"] = HOPS_UND[top_edges.i, top_edges.j]
top_edges["geo_km"] = GEO[top_edges.i, top_edges.j].round(2)
top_edges["train_corr"] = CORR[top_edges.i, top_edges.j].round(3)
top_edges.to_csv(os.path.join(OUT_DIR, "top_learned_edges.csv"), index=False)
print(top_edges.round(3).to_string(index=False))

# self-weight and "hub" structure of the learned graph
diag_w = np.diag(A_ADP)
in_strength = A_ADP.sum(0)
print(f"\nadaptive: mean self weight {diag_w.mean():.3f}; nodes with self in top-20: {(diag_w > 0).mean():.1%}")
print(f"adaptive: in-strength (column sums) max {in_strength.max():.1f}, top-10 hubs hold "
      f"{np.sort(in_strength)[-10:].sum() / N:.1%} of all weight; #nodes receiving any weight {(in_strength > 1e-6).sum()}")
# correlation between learned weight and training correlation among kept edges
r_, c_ = edge_set(A_ADP, CFG["topk"])
print("Spearman(learned weight, train corr) on kept edges:",
      round(pd.Series(A_ADP[r_, c_]).corr(pd.Series(CORR[r_, c_]), method="spearman"), 3))

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
gr = np.where(Abin); r1, c1 = edge_set(A_ADP, 1); r5, c5 = edge_set(A_ADP, 5)
bins = np.linspace(0, np.percentile(GEO, 99), 60)
for (r_, c_), lab, col in [(gr, "road-graph edges", "tab:blue"), ((r1, c1), "learned top-1", "tab:red"),
                           ((r5, c5), "learned top-5", "tab:orange"), ((rr[keep], rc[keep]), "random pairs", "tab:gray")]:
    ax[0].hist(GEO[r_, c_], bins=bins, density=True, histtype="step", lw=1.6, label=lab, color=col)
    ax[1].hist(CORR[r_, c_], bins=np.linspace(-0.2, 1, 60), density=True, histtype="step", lw=1.6, label=lab, color=col)
ax[0].set_xlabel("geographic distance (km)"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
ax[0].set_title("Q6(b) distance between connected sensors")
ax[1].set_xlabel("Pearson corr. of 2020 flow"); ax[1].set_title("flow correlation between connected sensors")
hop_cats = ["1", "2", "3", "4-5", "6+", "unreach."]
def hop_hist(r_, c_):
    h = HOPS_UND[r_, c_]
    return np.array([(h == 1).mean(), (h == 2).mean(), (h == 3).mean(), ((h >= 4) & (h <= 5)).mean(),
                     ((h >= 6) & np.isfinite(h)).mean(), np.isinf(h).mean()]) * 100
xb = np.arange(len(hop_cats)); wdt = 0.27
for o, (r_, c_), lab, col in [(-wdt, (r1, c1), "learned top-1", "tab:red"), (0, (r5, c5), "learned top-5", "tab:orange"),
                              (wdt, (rr[keep], rc[keep]), "random pairs", "tab:gray")]:
    ax[2].bar(xb + o, hop_hist(r_, c_), wdt, label=lab, color=col)
ax[2].set_xticks(xb); ax[2].set_xticklabels(hop_cats); ax[2].set_xlabel("hops in road graph (undirected)")
ax[2].set_ylabel("% of edges"); ax[2].legend(fontsize=8); ax[2].set_title("road-graph hop distance of learned edges")
savefig("q6b_learned_vs_physical.png"); plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
for a_, (r_, c_), t_ in [(ax[0], np.where(Abin), "road-distance graph (directed edges)"),
                         (ax[1], (r1, c1), "learned graph: strongest neighbour of each node")]:
    a_.scatter(sd_meta.Lng, sd_meta.Lat, s=4, c="k", zorder=3)
    phys = (Abin | Abin.T)[r_, c_]
    for i_, j_, p_ in zip(r_, c_, phys):
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
    if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.bool_): return bool(o)
    return o

summary = {
    "config": CFG, "hardware": HW, "smoke_test": SMOKE,
    "data": {"n_sensors": int(N), "t15": int(len(ts15)), "start": str(ts15[0]), "end": str(ts15[-1]),
             "missing_5min_frac": float(np.isnan(X5).mean()), "missing_15min_frac": float((~OBS).mean()),
             "filled_short": int(n_short), "filled_long": int(n_long),
             "train_mean": MU, "train_std": SD, "windows": {k: int(len(v)) for k, v in IDX.items()},
             "plot_sensors": KEEP, "selected_sensor_q3c": SEL,
             "q3c": {"one_hop": [int(sd_ids[j]) for j in one_hop], "two_hop_only": [int(sd_ids[j]) for j in two_only]}},
    "graph_stats": graph_stats,
    "results": {k: {kk: vv for kk, vv in v.items()} for k, v in RESULTS.items()},
    "naive_reference_test": {k: {"MAE": v[0], "RMSE": v[1]} for k, v in NAIVE.items()},
    "q1c_table": q1c_rows,
    "q2": {"a_ls": float(a_ls), "b_ls": float(b_ls)},
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(to_py(summary), f, indent=2, default=str)
print("saved results.json; files in", OUT_DIR, ":", sorted(os.listdir(OUT_DIR)))
print("figures:", sorted(os.listdir(FIG_DIR)))
