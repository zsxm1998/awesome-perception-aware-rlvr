# Data directory

Everything under `data/` except this file is git-ignored. Populate it with

```bash
bash scripts/prepare_data.sh all          # training / validation data for every method
bash scripts/prepare_eval_data.sh all     # evaluation benchmarks (data/eval/)
```

Layout:

```
data/
├── virl39k/train.parquet        # PAPO-processed ViRL39K
├── mmk12/test.parquet           # MMK12 test (validation during training)
├── geometry3k/{train,validation,test}.parquet
├── grit/{train,test}.parquet
├── deepeyes/train.parquet
└── eval/<benchmark>/...         # created by scripts/prepare_eval_data.sh
```

Use `DATA_ROOT=/path/to/data` to keep the data elsewhere; both the preparation scripts and
the training launchers honor it.

Downloads use the Hugging Face Hub. If `huggingface.co` is slow or blocked, set
`HF_ENDPOINT=https://hf-mirror.com`; the scripts then bypass any HTTP proxy for the mirror
(`HF_MIRROR_BYPASS_PROXY=0` disables this), because the mirror does not work through proxies.
GRIT images are fetched individually from the COCO and Visual Genome servers.
