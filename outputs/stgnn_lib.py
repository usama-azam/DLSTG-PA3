"""STGNN library for AI651 PA3 (models, training, evaluation, multi-GPU worker).

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
