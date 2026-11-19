# AI651 PA3 — Spatio-Temporal Traffic Forecasting on LargeST-SD

**Course:** AI651 Deep Learning for Space, Time and Graphs (LUMS, Fall 2026)
**Author:** Usama Azam (25280109)

This repo is Programming Assignment 3. It has two parts:

1. NumPy experiments that check what sum, mean and max aggregation can represent (Q1), and how a graph layer can tell which way a slowdown is moving on a ring road (Q2).
2. A lightweight spatio-temporal GNN (temporal convolution → graph convolution → MLP head) that forecasts 15-minute traffic flow for the **716 San Diego sensors of LargeST** (2020–2021), predicting the next 1 h from the last 2 h. It compares a temporal-only baseline with **fixed, adaptive (learned) and hybrid** adjacency matrices, plus four ablations, each trained with **3 seeds** (Q3–Q7).

Everything (all numbers and figures in the report) comes from **one notebook**, executed on Kaggle with 2 × Tesla T4 GPUs.

## Results (test: Apr–Dec 2021, vehicles per 15 min, mean ± std over seeds 42/43/44)

| Model | Adjacency | Val MAE | **Test MAE** | **Test RMSE** | Training time (T4) |
|---|---|---:|---:|---:|---:|
| Temporal-only | None | 55.68 ± 0.19 | 64.40 ± 0.14 | 101.32 ± 0.20 | 1407 s |
| STGNN | Fixed (road distance) | 49.31 ± 0.13 | 58.33 ± 0.15 | 91.93 ± 0.19 | 1692 s |
| STGNN | Adaptive (learned) | **41.65 ± 0.19** | **50.94 ± 0.36** | **80.60 ± 0.60** | 1790 s |
| STGNN | Hybrid (learned α → 0.05) | 42.22 ± 0.92 | 51.35 ± 0.72 | 81.15 ± 1.11 | 1807 s |

Naive references on the same test set: last value MAE 87.19, 2020 time-of-week mean MAE 134.20.

Ablations (test MAE):

| Ablation | Test MAE | Δ vs reference (better in) |
|---|---:|---|
| Fixed, 1 GC layer, no residual (the handout's form) | 137.58 ± 0.33 | +79.25 vs fixed (0/3) |
| Fixed, bidirectional graph D⁻¹(Ã+Ãᵀ) | 57.92 ± 0.07 | −0.41 vs fixed (3/3) |
| Temporal-only + 20-d node embedding | 60.15 ± 0.41 | −4.25 vs temporal-only (3/3) |
| Fixed + 20-d node embedding | 53.98 ± 0.25 | −4.36 vs fixed (3/3) |

Main findings:

- **Node identity explains most of the learned graph's advantage.** Per-sensor embeddings alone recover about 59% of the gap between the fixed and the adaptive graph. The remaining 3.03 MAE comes from the learned neighbour structure (better in 3/3 seeds).
- **The learned graph is not the road map.** A node's strongest learned neighbour is a road edge only 7.5% of the time, against 4.3% for random pairs, but it has a higher 2020 flow correlation than road neighbours (0.89 vs 0.855).
- **The specific learned edges are not stable across seeds.** The top-5 Jaccard overlap between seeds is only 0.014, even though the test error is almost identical across seeds.

See report §6 for the full analysis.

## Repository layout

```
├── notebook/
│   ├── dlstg_pa3.py                 # source of truth (jupytext "percent" format)
│   ├── dlstg_pa3.ipynb              # notebook generated from the .py (no outputs) — this is what runs on Kaggle
│   ├── stgnn_lib.py                 # models / training / evaluation (embedded in the notebook, which writes and runs it)
│   ├── dlstg_pa3_executed.ipynb     # the executed Kaggle run, with all outputs and figures
│   ├── dlstg_pa3_executed.html      # HTML render of the executed run
│   └── kernel-metadata.json         # Kaggle API metadata (dataset, GPU, privacy)
├── outputs/                         # everything the run wrote to /kaggle/working/outputs
│   ├── figures/*.png                # all report figures
│   ├── results.json                 # config, hardware, timing, data/fault stats, summary, paired tests, graph analysis
│   ├── results_summary.csv          # mean ± std per configuration (overall + per horizon)
│   ├── results_per_seed.csv         # every run (8 configurations × 3 seeds)
│   ├── paired_comparisons.csv       # per-seed paired differences of test MAE
│   ├── graph_analysis.csv           # learned-vs-physical graph statistics (mean ± std over seeds)
│   ├── graph_analysis_per_seed.csv
│   ├── top_learned_edges.csv        # 15 strongest learned connections (seed 42)
│   ├── runs/                        # per run: history (run_*.json), weights (model_*.pt),
│   │                                #   learned adjacency (A_*.npy), test predictions (pred_*_s42.npy), worker logs
│   ├── stgnn_lib.py                 # copy of the library that produced the runs
│   └── kaggle_run_log.txt           # full stdout of the Kaggle run
├── outputs.zip                      # zip of outputs/
├── report/
│   ├── report.tex, theory.tex       # LaTeX source of the report
│   ├── figures/                     # copies of outputs/figures used by the report
│   └── report.pdf                   # compiled report
└── requirements.txt
```

## Reproducing the results

Seeds: **42, 43, 44** for every configuration. Hyper-parameters are in the `CFG` dict in the first code cell and in report §7.
The notebook launches one worker process per visible GPU (it also works with one GPU, or on a CPU, just more slowly).
On Kaggle with 2 × T4 the full run (24 trainings) takes about **5.7 hours**, well within Kaggle's 12-hour limit.

### Option A: Kaggle via the API (how the submitted run was made)

1. Install the CLI and add your token: `pip install kaggle`, then put `kaggle.json` in `~/.kaggle/` (or `%USERPROFILE%\.kaggle\` on Windows).
2. Edit `notebook/kernel-metadata.json` and set `"id"` to `<your-kaggle-username>/dlstg-pa3-largest-sd-stgnn`.
3. Push and run. This is a "Save & Run All" commit run on GPU (Kaggle attaches 2 × T4):
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

Upload `notebook/dlstg_pa3.ipynb` (**New Notebook → File → Import**), add the dataset *LargeST* (`liuxu77/largest`) as input, choose the *GPU T4 x2* accelerator, then **Save Version → Save & Run All**.

### Option C: locally

```bash
pip install -r requirements.txt
kaggle datasets download liuxu77/largest -f ca_meta.csv -p data/largest
kaggle datasets download liuxu77/largest -f ca_rn_adj.npy -p data/largest
kaggle datasets download liuxu77/largest -f ca_his_raw_2020.h5 -p data/largest
kaggle datasets download liuxu77/largest -f ca_his_raw_2021.h5 -p data/largest
# unzip any downloaded *.zip in data/largest, then:
export LARGEST_DIR=data/largest      # PowerShell: $env:LARGEST_DIR="data/largest"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebook/dlstg_pa3.ipynb --output dlstg_pa3_local.ipynb
```

The two `.h5` files are about 7.2 GB each. They are read with `h5py` in row blocks, and the notebook process peaked at 3.4 GB of RAM. Outputs go to `outputs/`.

If the dataset is not found, the notebook switches to a **synthetic smoke test** (60 fake sensors, 1 epoch, all 3 seeds and all 8 configurations), which runs the whole pipeline, including the multi-seed analysis, in about 10–15 minutes on a CPU.

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
  - Runs of ≥ 1 h of zero flow are detector faults and are set to missing.
  - Missing targets are masked out of the loss and all metrics.
  - Missing inputs are imputed **causally**: gaps up to 1 h are forward-filled; longer gaps use the sensor's 2020 time-of-week profile.
  - Sensors with < 50% observed 2020 data (26 of 716) stay as targets but are never used as graph message sources.
  - Splits: train 2020; validation Jan–Mar 2021; test Apr–Dec 2021. No window crosses a split boundary, and z-score statistics come from 2020 only.
- **Model (Q4):** shared temporal encoder (Conv1d k=3 → Conv1d k=3 dilation 2 → Linear, d=32), then 2 residual graph-conv layers `H ← H + ReLU(Â H W)`, then a per-node MLP head that outputs 4 horizons.
  - Loss: masked MAE.
  - Optimiser: Adam with lr 1e-3, batch 64, ReduceLROnPlateau (×0.5, patience 3).
  - Up to 80 epochs, with early stopping (patience 10).
- **Graphs (Q5):**
  - *Fixed:* directed random-walk normalisation `D⁻¹Ã` of the road-distance adjacency, which is not symmetric.
  - *Adaptive:* `softmax_row(topk₂₀(ReLU(E₁E₂ᵀ)))` with d_e = 10. Sparsification happens before the softmax, and gradients flow through the kept logits.
  - *Hybrid:* `α·Â_fixed + (1−α)·A_adaptive`, with a learnable α = sigmoid(a). Both terms are row-stochastic.
- **Ablations:**
  - the handout's single GC layer without residual;
  - a bidirectional fixed graph;
  - a per-sensor embedding added to the temporal-only and fixed models, which separates node identity from graph structure.

## Acknowledgements

Dataset: LargeST, Liu et al., NeurIPS 2023 Datasets & Benchmarks ([GitHub](https://github.com/liuxu77/LargeST)).
Generative-AI use is disclosed in the report appendix, as required by the course policy.
