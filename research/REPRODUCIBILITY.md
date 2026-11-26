# Reproducibility and provenance

The public archive can reproduce the **reported test metrics** from predictions on a CPU. An exact training rerun also requires authorized access to the complete study checkout and the FP16 cache or model artifacts. The workflow below records the commands used by that checkout.

## Verify the archive

From the public repository root, use Python 3.11 or later.

```bash
python -m llmembed_research.results --output outputs/r52_summary.csv
```

This recomputes Accuracy, Macro-F1, Rare F1, sample standard deviations, and paired differences against AG-CMF + CE. The test labels and example order must agree across nine runs. The verifier fixes the rare classes from the archived train counts, excludes the six documented duplicates for the filtered calculation, and compares each run with the recorded metrics. The per-seed CSVs and the predictions use integer labels and stable test IDs. [Verification notebook](../notebooks/verify_r52.ipynb) · [Public test results](audit/r52_confirmation/study_metrics.csv) · [Per-class summary](audit/r52_confirmation/per_class_summary.csv)

Install PyTorch to run the objective and gate tests. The code in `llmembed_research/` exposes the study-specific operations for use with a separately obtained model head.

```bash
python -m pip install -r requirements-research.txt
python -m unittest discover -s tests -v
```

## Data, extraction, and training config

R52 uses the [TextGCN corpus and source split](https://github.com/yao8839836/text_gcn/tree/962223652e9bb164ac2d83cd09fc7b8845ce860b/data), derived from Reuters-21578, Distribution 1.0. The loader preserves input text and source test order, maps labels consistently, and takes a class-stratified validation subset from source train with seed 42 and a target fraction of 0.1. The resulting sizes are 5,870 train, 662 validation, and 2,568 test. Validation covers 51 of 52 classes. Frequency groups are defined by train support of 1–9, 10–49, and at least 50. The source split is retained for the primary comparison, with exact train-text matches excluded in sensitivity analysis.

| Backbone | Pinned model revision | Stored representation |
|---|---|---|
| `meta-llama/Llama-2-7b-hf` | `01c7f73d771dfac7d292323805ebc428287df4f9` | Hidden layers 32, 28, 24, 20, 16 with prompt-based token pooling |
| `google-bert/bert-large-uncased` | `6da4b6a26a1877e173fca3225479512db81a5e5b` | Layer 24 pooler output |
| `FacebookAI/roberta-large` | `722cf37b1afa9454edce342e7895e588b6ff1d59` | Layer 24 first-token representation |

Extract with Transformers 4.51.3, FP16, seed 42, and windows of 256 content tokens with 32-token overlap. The LLaMA token pooling parameters are γ = 0.5 and τ = 0.1. The separate training loss uses logit-adjustment τ = 1. The confirmed environment was Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, and two Tesla T4 GPUs. The downstream head used AdamW at learning rate 1e-4 with weight decay 0.01, cosine scheduling, effective batch 1,024, microbatch 64, hidden width 1,024, eight attention heads, and σ = 0.1. Training runs use 100 epochs and seeds 42, 43, and 44, with the first epoch at maximum validation accuracy selected. [Recorded config](../configs/r52.json) · [Extraction receipt](audit/kaggle_extraction_v2/r52_fp16/cache_receipt.json) · [Confirmation manifest](audit/kaggle_downstream_v9/study/manifests/confirm-1608e056c98bf7c8.json)

## Commands for an authorized training checkout

Use the complete local research checkout containing `llmembed.study` at the executed code revision `782dca953c7f180977f82cb59867287dd1c92f93`. The permissioned source archive includes `kaggle_experiment/requirements.txt`. Keep the GPU-compatible PyTorch build on Kaggle and install the remaining experiment dependencies from that file. Set `STUDY_CHECKOUT`, `FEATURES_DIR`, `CACHE_RECEIPT`, and `R52_DECISION` to the appropriate local paths. The decision file records a validation-only method choice. [Frozen selection](audit/r52_pilot/decision.json)

```bash
cd "$STUDY_CHECKOUT/kaggle_experiment"
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m llmembed.study audit --dataset r52 --output runs/r52-audit

# If a compatible cache is available, start here. The receipt records hashes for all nine stores.
python -m llmembed.study check-cache --dataset r52 --output runs/r52-cache-check \
  --reuse-cache "$FEATURES_DIR" --receipt "$CACHE_RECEIPT"

# These are separate validation jobs at seed 42.
python -m llmembed.study pilot --dataset r52 --experiment exp4 --output runs/r52-pilot \
  --reuse-cache "$FEATURES_DIR" --receipt "$CACHE_RECEIPT"
python -m llmembed.study pilot --dataset r52 --experiment imbalance_hierarchical_ce --output runs/r52-pilot \
  --reuse-cache "$FEATURES_DIR" --receipt "$CACHE_RECEIPT"
python -m llmembed.study pilot --dataset r52 --experiment imbalance_flat_logit_adjustment --output runs/r52-pilot \
  --reuse-cache "$FEATURES_DIR" --receipt "$CACHE_RECEIPT"

# Confirm three heads across seeds 42/43/44 using the validated pilot decision.
python -m llmembed.study confirm --dataset r52 --output runs/r52-confirm \
  --reuse-cache "$FEATURES_DIR" --receipt "$CACHE_RECEIPT" --decision "$R52_DECISION"
```

To register a compatible existing FP16 cache, use `llmembed.study lock-cache` with `--reuse-cache`, `--receipt`, and a versioned `--origin`. For a new cache, run the following once with a CUDA-enabled environment and authorized access to the gated LLaMA-2 revision. Provide model access through Kaggle Secrets or the local authentication environment.

```bash
python -m llmembed.study extract-r52 --dataset r52 --output runs/r52-fp16 \
  --origin "<versioned extraction run>" --code-commit 782dca953c7f180977f82cb59867287dd1c92f93
```

The extraction produces `runs/r52-fp16/features/` and its cache receipt. All downstream comparisons should reuse that same cache. Frozen backbones provide features. Chunk pooling and the classification head are trainable. [Kaggle extraction v2](https://www.kaggle.com/code/longvucs/llmembed-fusion-r52-fp16-extraction) · [Kaggle downstream v6–v9](https://www.kaggle.com/code/longvucs/llmembed-fusion-fp16-downstream)

## Publication scope

The [ACL 2024 implementation](https://github.com/ChunLiu-cs/LLMEmbed-ACL2024) and [teammate implementation](https://github.com/ThanhNosaur/LLMEmbed) had no discoverable repository license at release. This public archive publishes the study-specific operations and independently checked numeric artifacts without carrying their source history or bundled GPU notebook code. Upstream copyrights and terms continue to apply. The complete research branch and original artifacts are retained in the permissioned workspace. The public repository does not bundle Reuters text, model weights, feature stores, checkpoints, access credentials, or raw runtime logs. [GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
