# LLM Playground

LLM Playground is an end-to-end sandbox for training and sampling decoder-only
Transformers on custom corpora. It bundles together data preprocessing,
dataset + dataloader utilities, and a transparent PyTorch training loop that
prioritizes checkpointing + resuming so you can iterate on research ideas
without re-building the same scaffolding.

## Highlights

- **Config-first experiments.** Hydra drives every stage (preprocessing,
  dataset creation, optimization, sampling) with custom resolvers registered in
  `playground.__init__` so relative paths like `${root:data/...}` just work.
- **Grokable GPT-style model.** `playground.transformer.Transformer` implements a
  GPT-2 sized architecture with tied/untied heads, dropout controls, learned
  position embeddings, KV caching, and batched autoregressive decoding.
- **Reusable trainer.** `playground.trainer.Trainer` wires together datasets,
  optimizers, schedulers, logging, checkpoint dirs, and text sampling
  without hiding the PyTorch training loop.
- **Data helpers.** `scripts/preprocess_pretraing_data.py` tokenizes raw text,
  performs train/val splits per shard, and writes them to
  `data/processed/{train,validation}`.
- **Token-wise datasets.** `playground.dataloader.NextTokenPredictionDataset`
  slices contiguous token windows with configurable `max_length` and `stride`
  so you can switch between overlapping and disjoint chunks.
- **Hydra-friendly logging + sampling.** The trainer periodically evaluates,
  saves checkpoints, and (optionally) emits greedy generations from prompt lists
  via `configs/experiments/pretraining/trainer/sampling/*`.

## Repository layout

```
├── configs/
│   ├── preprocessing/             # Pre-tokenisation + splitting
│   └── experiments/pretraining/   # Modular Hydra configs (model, data, trainer)
├── data/
│   ├── raw/                       # Drop your source .txt shards here
│   └── processed/                 # Train/val splits created by the preprocess script
├── scripts/
│   ├── preprocess_pretraing_data.py
│   └── pretrain.py
├── src/playground/
│   ├── transformer.py             # GPT-style model with KV cache decoding
│   ├── trainer.py                 # Training loop + callbacks
│   ├── dataloader.py              # Token chunking datasets + loaders
│   ├── inference_utils.py         # Sampling helpers, caches, masking
│   └── ...                        # Losses, utils, logging, metadata
└── test/
    └── test_mha_fused.py          # Example unit test for fused attention
```

## Getting started

### Prerequisites

- Python 3.11+
- A CUDA-capable GPU (trainer auto-selects `cuda` when available but also works
  on CPU for experiments)
- `uv` for dependency management

### Installation

```bash
# clone the repo
cd llm-playground

# create a virtual environment (uv example)
uv venv --python 3.11
source .venv/bin/activate

# install the project in editable mode
uv pip install -e .
```

> 💡 If you wish to contribute, use instead 
> `uv pip install -e ".[dev]"` which will also install `pytest` and `pre-commit`.

### Running the tests

The repository currently ships with a fused multi-head attention test. If you installed with the
`dev` option you can run the whole suite with:

```bash
pytest
```

## Data preprocessing

1. Drop your raw `.txt` shards into `data/raw/`.
2. Edit `configs/preprocessing/pretrain_preprocess.yaml` to point to the proper
   directories or adjust the train/validation split ratio.
3. Execute the preprocessing script (Hydra will resolve output dirs via the
   `${root:...}` resolver):

```bash
python scripts/preprocess_pretraing_data.py input_dir=data/raw out_dir=data/processed
```

Each shard is tokenised with GPT-2 BPE, split into train/validation text, and
written to `data/processed/{train,validation}/<shard>.txt` for downstream use.

## Training workflow

1. Set `trainer.experiment_name` inside
   `configs/experiments/pretraining/trainer/pretrain_verdict.yaml` so checkpoints
   land under `models/<experiment>/<timestamp>_seed_<seed>/`.
2. (Optional) customise sub-configs:
   - `model/*` – depth, width, dropout, tied embeddings, context length.
   - `optimiser/*` – optimizer hyperparameters and warmup/cosine schedule.
   - `dataset/*` – dataset paths, token window length, stride, etc.
   - `trainer/sampling/*` – prompt list, sample cadence, max generated tokens.
3. Launch training:

```bash
python scripts/pretrain.py \
    trainer.experiment_name=my_experiment \
    trainer.num_epochs=5 \
    optimiser.optimiser.lr=3e-4
```

During training the `Trainer` handles:

- Deterministic seeding + device selection via `trainer.device` (`auto`, `cuda`,
  `cpu`).
- Separate logging/validation/save cadences (`log_steps`, `eval_steps`,
  `save_steps`).
- Weight-decay aware parameter grouping (norms/embeddings are excluded).
- Scheduler warmup steps derived from total training steps.
- Optional prompt sampling every N steps or per epoch with optional disk dumps.

## Inference & sampling

After training, load your checkpoint through Hydra or manually instantiate the
model + tokenizer. `playground.transformer.Transformer.generate` supports
batched decoding with or without KV caching, early EOS termination, and
truncation guards. The sampling helpers in `playground.inference_utils` expose
composable top-k, temperature, and greedy decoders.

Example snippet:

```python
from playground.transformer import Transformer
from playground.inference_utils import greedy_decode
import tiktoken
import torch

cfg = ...  # same cfg used during training
model = Transformer(cfg)
model.load_state_dict(torch.load("path/to/checkpoint.pt"))
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
prompt_ids = torch.tensor([tokenizer.encode("Every effort moves you")])
output_ids = model.generate(prompt_ids, max_new_tokens=64, eos_token_id=tokenizer.eot_token)
print(tokenizer.decode(output_ids[0].tolist()))
```

## Extending the playground

- Add new datasets by creating another YAML under
  `configs/experiments/pretraining/dataset/` and pointing to your processed
  files.
- Swap in different optimizers or schedulers by dropping configs in the
  respective folders and overriding via CLI (`optimiser=adamw_large_batch`).
- Implement new sampling strategies by subclassing `playground.logit_processors`
  or tweaking `trainer/sampling` configs.
- Use `playground.utils` and `playground.trainer_utils` for reproducibility
  helpers (deterministic seeds, device moves, token counting, etc.).

## Roadmap ideas

- Evaluation harness for downstream tasks (perplexity, QA, etc.).
- Mixed-precision + gradient accumulation utilities.
- More loggers (TensorBoard, Weights & Biases) wired into the Trainer.
- Dataset streaming + shuffling for multi-billion token corpora.

---

Happy hacking! If you build something neat with the playground or have ideas to
improve the ergonomics, feel free to open an issue or PR.
