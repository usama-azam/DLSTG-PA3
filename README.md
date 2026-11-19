# AI651 PA3 — Spatio-Temporal Traffic Forecasting on LargeST-SD

**Course:** AI651 Deep Learning for Space, Time and Graphs (LUMS, Fall 2026)
**Author:** Usama Azam (25280109)

This repo is Programming Assignment 3. It has two parts:

1. NumPy experiments that check what sum, mean and max aggregation can represent (Q1), and how a graph layer can tell which way a slowdown is moving on a ring road (Q2).
2. A lightweight spatio-temporal GNN (temporal convolution → graph convolution → MLP head) that forecasts 15-minute traffic flow for the **716 San Diego sensors of LargeST** (2020–2021), predicting the next 1 h from the last 2 h. It compares a temporal-only baseline with **fixed, adaptive (learned) and hybrid** adjacency matrices (Q3–Q7).

Everything (all numbers and figures in the report) comes from **one notebook**, executed on Kaggle with a Tesla T4 GPU.

## Results (test: Apr–Dec 2021, vehicles per 15 min)

| Model | Adjacency | Val MAE | Val RMSE | **Test MAE** | **Test RMSE** | Training time (T4) |
|---|---|---:|---:|---:|---:|---:|
| Temporal-only | None | 55.72 | 89.61 | 64.23 | 100.95 | 842 s |
| STGNN | Fixed (road distance) | 50.06 | 80.63 | 58.88 | 92.66 | 978 s |
| STGNN | Adaptive (learned) | **42.14** | **68.10** | **51.34** | **81.17** | 1063 s |
| STGNN | Hybrid (learned α → 0.057) | 42.33 | 68.75 | 51.43 | 81.39 | 1067 s |

Naive references on the same test set: last value MAE 87.14, 2020 time-of-week mean MAE 134.27.

Each node's strongest learned neighbour is a physical road edge only 7.4% of the time, against 4.1% for random pairs. It does, however, have a higher 2020 flow correlation than road neighbours (0.90 vs 0.85). In other words, the model learns a *functional* similarity graph rather than the road map. See report §6.

## Repository layout

```
├── notebook/
│   ├── dlstg_pa3.py                 # source of truth (jupytext "percent" format)
│   ├── dlstg_pa3.ipynb              # notebook generated from the .py (no outputs) — this is what runs on Kaggle
│   ├── dlstg_pa3_executed.ipynb     # the executed Kaggle run, with all outputs and figures
│   ├── dlstg_pa3_executed.html      # HTML render of the executed run
│   └── kernel-metadata.json         # Kaggle API metadata (dataset, GPU, privacy)
├── outputs/                         # everything the run wrote to /kaggle/working/outputs
│   ├── figures/*.png                # all report figures
│   ├── results.json                 # config, hardware, data stats, per-model metrics + training history
│   ├── results_table.csv            # Q6(a) comparison table (incl. per-horizon metrics)
│   ├── graph_analysis.csv           # Q6(b) learned-vs-physical graph statistics
│   ├── top_learned_edges.csv        # 15 strongest learned connections
│   ├── A_fixed_normalized.npy       # D^-1 Ã (716×716)
│   ├── A_adaptive.npy / A_hybrid.npy# learned adjacency matrices (716×716)
│   ├── model_{none,fixed,adaptive,hybrid}.pt   # trained weights (best validation epoch)
│   └── kaggle_run_log.txt           # full stdout of the Kaggle run
├── outputs.zip                      # zip of outputs/
├── report/
│   ├── report.tex, theory.tex       # LaTeX source of the report
│   ├── figures/                     # copies of outputs/figures used by the report
│   └── report.pdf                   # compiled report
└── requirements.txt
```

## Reproducing the results

Fixed seed: **42**. Hyper-parameters are in the `CFG` dict in the first code cell and in report §7.
GPU is optional; on a T4 the whole notebook takes about 70 minutes.

### Option A: Kaggle via the API (how the submitted run was made)

1. Install the CLI and add your token: `pip install kaggle`, then put `kaggle.json` in `~/.kaggle/` (or `%USERPROFILE%\.kaggle\` on Windows).
2. Edit `notebook/kernel-metadata.json` and set `"id"` to `<your-kaggle-username>/dlstg-pa3-largest-sd-stgnn`.
3. Push and run. This is a "Save & Run All" commit run on a T4:
   ```bash
   cd notebook
   kaggle kernels push -p . --accelerator NvidiaTeslaT4
   kaggle kernels status <your-kaggle-username>/dlstg-pa3-largest-sd-stgnn
   ```
4. When the status is `COMPLETE`, download the outputs:
   ```bash
   kaggle kernels output <your-kaggle-username>/dlstg-pa3-largest-sd-stgnn -p ../kaggle_run
   ```
   The executed `.ipynb` is not part of the API output. Download it from the notebook page (**File → Download notebook**), or attach the notebook as an input to another kernel and copy `__notebook__.ipynb` out.

The metadata attaches the dataset [`liuxu77/largest`](https://www.kaggle.com/datasets/liuxu77/largest), which the notebook finds at `/kaggle/input/datasets/liuxu77/largest` (older mount paths are also checked).

### Option B: Kaggle web UI

Upload `notebook/dlstg_pa3.ipynb` (**New Notebook → File → Import**), add the dataset *LargeST* (`liuxu77/largest`) as input, choose the *GPU T4* accelerator, then **Save Version → Save & Run All**.

### Option C: locally

```bash
pip install -r requirements.txt
kaggle datasets download liuxu77/largest -f ca_meta.csv -p data/largest
kaggle datasets download liuxu77/largest -f ca_rn_adj.npy -p data/largest
kaggle datasets download liuxu77/largest -f ca_his_raw_2020.h5 -p data/largest
kaggle datasets download liuxu77/largest -f ca_his_raw_2021.h5 -p data/largest
# unzip any downloaded *.zip in data/largest, then:
export LARGEST_DIR=data/largest      # PowerShell: $env:LARGEST_DIR="data/largest"
jupyter nbconvert --to notebook --execute notebook/dlstg_pa3.ipynb --output dlstg_pa3_local.ipynb
```

The two `.h5` files are about 7.2 GB each. They are read with `h5py` in row blocks, so about 4 GB of RAM is enough. Outputs go to `outputs/`.

If the dataset is not found, the notebook switches to a **synthetic smoke test** (60 fake sensors, 2 epochs), which checks the whole pipeline in a few minutes on a CPU.

### Editing the notebook

`notebook/dlstg_pa3.py` is the source. After editing it, regenerate the notebook:

```bash
jupytext --to ipynb --set-kernel python3 --output notebook/dlstg_pa3.ipynb notebook/dlstg_pa3.py
```

### Building the report

```bash
cd report && latexmk -pdf report.tex      # or: tectonic report.tex / upload the folder to Overleaf
```

## Method summary

- **Data (Q3):** San Diego = Caltrans District 11 in `ca_meta.csv` (716 sensors).
  - PeMS flow is a per-interval vehicle count, so 5-min readings are **summed** into 15-min intervals (vehicles per 15 min).
  - Missing targets are masked out of the loss and all metrics.
  - Missing inputs: gaps up to 1 h are linearly interpolated; longer gaps are filled with the sensor's 2020 time-of-week profile.
  - Splits: train 2020; validation Jan–Mar 2021; test Apr–Dec 2021. No window crosses a split boundary, and z-score statistics come from 2020 only.
- **Model (Q4):** shared temporal encoder (Conv1d k=3 → Conv1d k=3 dilation 2 → Linear, d=32), then 2 residual graph-conv layers `H ← H + ReLU(Â H W)`, then a per-node MLP head that outputs 4 horizons. Loss is masked MAE; Adam with lr 1e-3, batch 64; up to 50 epochs with early stopping (patience 8).
- **Graphs (Q5):**
  - *Fixed:* directed random-walk normalisation `D⁻¹Ã` of the road-distance adjacency, which is not symmetric.
  - *Adaptive:* `softmax_row(topk₂₀(ReLU(E₁E₂ᵀ)))` with d_e = 10. Sparsification happens before the softmax, and gradients flow through the kept logits.
  - *Hybrid:* `α·Â_fixed + (1−α)·A_adaptive`, with a learnable α = sigmoid(a). Both terms are row-stochastic.

## Acknowledgements

Dataset: LargeST, Liu et al., NeurIPS 2023 Datasets & Benchmarks ([GitHub](https://github.com/liuxu77/LargeST)).
Generative-AI use is disclosed in the report appendix, as required by the course policy.
