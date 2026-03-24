# Embedding Module

This module runs the embedding phase as a separate step after chunking.

The embedding flow is settings-driven:

1. Read the active chunking strategy from `config/appsettings.json`.
2. Load `05_chunks.json` files only for that strategy.
3. Build the final text sent to the embedding provider.
4. Generate vectors in batches.
5. Persist embedding records and a run manifest.
6. Optionally export a Spotlight dataset for inspection.

## Prerequisites

- Python dependencies installed from `requirements.txt`
- Chunk outputs already generated under `data/chunks`
- `config/appsettings.json` configured for the intended strategy and embedding settings
- `OPENAI_API_KEY` available in the environment when using the OpenAI provider

Optional:

- Renumics Spotlight installed when using the visualization helpers

## Required environment variables

### `OPENAI_API_KEY`

Required by `embedding/providers/openai_provider.py` when `embedding.provider` is set to `openai`.

Example:

```bash
export OPENAI_API_KEY="your_api_key"
```

## Central configuration

The module reads its runtime configuration from `config/appsettings.json`.

Current example:

```json
{
  "chunking": {
    "strategy": "article_smart"
  },
  "embedding": {
    "enabled": false,
    "provider": "openai",
    "model": "text-embedding-3-large",
    "input_root": "data/chunks",
    "output_root": "data/embeddings",
    "input_text_field": "text_for_embedding",
    "batch_size": 100,
    "visualization": {
      "enabled": false,
      "spotlight_enabled": false
    }
  }
}
```

Relevant keys:

- `chunking.strategy`: strategy used by chunking and later consumed by embedding
- `embedding.enabled`: must be `true` before running `python3 -m embedding.main`
- `embedding.provider`: current provider selector used by the factory
- `embedding.model`: embedding model name sent to the provider
- `embedding.input_root`: root folder where chunk outputs are discovered
- `embedding.output_root`: root folder where embedding artifacts are written
- `embedding.input_text_field`: preferred chunk field used as embedding input
- `embedding.batch_size`: maximum records sent per provider batch
- `embedding.visualization.enabled`: enables visualization export during indexing
- `embedding.visualization.spotlight_enabled`: enables Spotlight dataset export

## How chunking and embedding connect

Chunking and embedding are separate phases.

The embedding loader searches for files named `05_chunks.json` under:

```text
data/chunks/<doc_id>/<strategy>/05_chunks.json
```

Only the strategy configured in `chunking.strategy` is accepted during embedding input loading.

## Run chunking

Chunking is executed through the main pipeline entrypoint and uses the strategy configured in `config/appsettings.json`.

```bash
python3 src/main.py
```

This generates chunk outputs under `data/chunks`.

## Run embedding

Before running embedding:

- set `embedding.enabled` to `true`
- ensure chunk outputs already exist for the configured strategy
- export `OPENAI_API_KEY`

Run:

```bash
python3 -m embedding.main
```

The entrypoint validates that embedding is enabled, executes the indexer, and prints a short summary with the generated artifact paths.

## Output structure

Embedding outputs are written under:

```text
data/embeddings/<strategy>/<run_id>/
```

Artifacts created by the storage layer:

- `embedding_records.json`
- `run_manifest.json`

When Spotlight export is enabled, the run directory also contains:

- `spotlight_dataset.jsonl`

## Spotlight visualization

To generate the Spotlight dataset during embedding execution, enable both visualization flags in `config/appsettings.json`:

```json
"visualization": {
  "enabled": true,
  "spotlight_enabled": true
}
```

Then run:

```bash
python3 -m embedding.main
```

To open the most recent exported dataset in Spotlight:

```bash
python3 -m embedding.visualization.spotlight_viewer
```

To open a specific dataset:

```bash
python3 -m embedding.visualization.spotlight_viewer \
  --strategy article_smart \
  --run-id <run_id>
```

The viewer expects a `spotlight_dataset.jsonl` file and requires a working Spotlight installation.
