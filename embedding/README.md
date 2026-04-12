# Embedding Module

This module runs the embedding phase after chunking.

The current main flow is:

1. Read the active chunking and embedding settings from `config/appsettings.json`.
2. Load `05_chunks.json` only for the configured chunking strategy.
3. Build the final embedding text for each chunk.
4. Generate vectors with the configured provider and model.
5. Persist the vectors to ChromaDB as the main storage.
6. Write local auxiliary artifacts for audit and visualization support.

## Prerequisites

- Python dependencies installed from `requirements.txt`
- Chunk outputs already generated under `data/chunks`
- `config/appsettings.json` configured for the intended chunking and embedding run
- ChromaDB runtime configuration present for the selected mode

Optional:

- Renumics Spotlight installed when using the visualization helpers

## Required dependencies

The main embedding flow depends on:

- `sentence-transformers` for local embedding generation
- `chromadb` for embedding persistence

The OpenAI package can still exist in the repository because the provider factory
still supports that provider, but `OPENAI_API_KEY` is no longer required for the
main configured flow.

## Required environment variables

### `CHROMA_API_KEY`

Required only when `embedding.chromadb.mode` is set to `cloud`.

The variable name is controlled by:

```json
"embedding": {
  "chromadb": {
    "cloud": {
      "api_key_env_var": "CHROMA_API_KEY"
    }
  }
}
```

When `embedding.chromadb.mode` is `persistent`, no ChromaDB cloud API key is required.

### `OPENAI_API_KEY`

Not required for the main embedding flow.

It is only needed when `embedding.provider` is explicitly switched to `openai`.

## Central configuration

The module reads its runtime configuration from `config/appsettings.json`.

Current example:

```json
{
  "chunking": {
    "strategy": "article_smart"
  },
  "embedding": {
    "enabled": true,
    "provider": "sentence_transformers",
    "model": "all-MiniLM-L6-v2",
    "input_root": "data/chunks",
    "output_root": "data/embeddings",
    "input_text_field": "text",
    "batch_size": 100,
    "chromadb": {
      "mode": "cloud",
      "persist_directory": "data/chromadb",
      "collection_name": "rag_embeddings",
      "cloud": {
        "tenant": "",
        "database": "",
        "api_key_env_var": "CHROMA_API_KEY"
      }
    },
    "visualization": {
      "enabled": true,
      "spotlight_enabled": true
    }
  }
}
```

Relevant keys:

- `chunking.strategy`: strategy used by chunking and later consumed by embedding
- `embedding.enabled`: must be `true` before running `python3 -m embedding.main`
- `embedding.provider`: provider selector used by the provider factory
- `embedding.model`: embedding model name used by the configured provider
- `embedding.input_root`: root folder where chunk outputs are discovered
- `embedding.output_root`: root folder where local embedding artifacts are written
- `embedding.input_text_field`: preferred chunk field used as embedding input
- `embedding.batch_size`: maximum records sent per provider batch
- `embedding.chromadb.mode`: ChromaDB target mode, currently `cloud` or `persistent`
- `embedding.chromadb.persist_directory`: local ChromaDB path used in persistent mode
- `embedding.chromadb.collection_name`: ChromaDB collection used by the storage layer
- `embedding.chromadb.cloud.tenant`: ChromaDB cloud tenant used in cloud mode
- `embedding.chromadb.cloud.database`: ChromaDB cloud database used in cloud mode
- `embedding.chromadb.cloud.api_key_env_var`: environment variable name read for cloud authentication
- `embedding.visualization.enabled`: enables visualization export during indexing
- `embedding.visualization.spotlight_enabled`: enables Spotlight dataset export

Cloud account setup is outside the scope of this module documentation. The
module only expects the runtime settings and environment variables required to
connect when cloud mode is selected.

## How chunking and embedding connect

Chunking and embedding remain separate phases.

The embedding loader searches for files named `05_chunks.json` under:

```text
data/chunks/<doc_id>/<strategy>/05_chunks.json
```

Only the strategy configured in `chunking.strategy` is accepted during
embedding input loading. This keeps embedding aligned with the active chunking
strategy selected in settings.

## Run chunking

Chunking is executed through the main pipeline entrypoint and uses the strategy
configured in `config/appsettings.json`.

```bash
python3 src/main.py
```

This generates chunk outputs under `data/chunks`.

## Run embedding

Before running embedding:

- set `embedding.enabled` to `true`
- ensure chunk outputs already exist for the configured strategy
- confirm `embedding.provider`, `embedding.model`, and `embedding.chromadb.*` are configured correctly
- export `CHROMA_API_KEY` only if `embedding.chromadb.mode` is `cloud`

Run:

```bash
python3 -m embedding.main
```

The entrypoint validates that embedding is enabled, runs the indexer, persists
embeddings to ChromaDB, and prints a short summary with the provider, model,
ChromaDB target, and generated artifact paths.

## Troubleshooting local Sentence Transformers startup

If `python3 -m embedding.main` fails with an error mentioning
`libtorchcodec`, `TorchCodec`, or missing CUDA libraries, the issue is in the
local Python runtime rather than in the embedding orchestration code.

Typical causes:

- `torch` was installed with a CUDA build, but the required CUDA runtime
  libraries are not available on the machine
- `torchcodec` is installed in the same environment and is incompatible with
  the local `torch` build
- FFmpeg runtime libraries required by `torchcodec` are missing

Recommended fixes:

- use a clean virtual environment with a CPU-only PyTorch installation when GPU
  acceleration is not required for embeddings
- remove `torchcodec` from the environment if the project does not intentionally
  use media decoding features
- reinstall `torch`, `sentence-transformers`, and related packages together in
  the same virtual environment so the runtime stack is aligned

The provider now reports the detected versions of `torch`, `torchcodec`,
`sentence-transformers`, and `transformers` in the startup error to make
environment mismatches easier to diagnose.

## What ChromaDB is used for

ChromaDB is the main persistence layer for embedding vectors in this module.

The storage layer uses it to:

- create or reuse the configured collection
- replace previously stored records for the active strategy
- upsert the new embedding vectors and metadata

The local output directory is not the source of truth for stored vectors. It is
used for run-level audit artifacts and optional visualization exports.

## Local auxiliary artifacts

Local artifacts are written under:

```text
data/embeddings/<strategy>/<run_id>/
```

Each new embedding execution replaces the previously persisted local run folder
for the same strategy before writing the next one.

Artifacts created by the embedding flow:

- `chromadb_storage.json`: local audit summary of the ChromaDB persistence result
- `run_manifest.json`: run-level manifest describing the completed embedding execution
- `spotlight_dataset.jsonl`: optional Spotlight dataset export when visualization is enabled

## Spotlight visualization

To generate the Spotlight dataset during embedding execution, enable both
visualization flags in `config/appsettings.json`:

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

To open a specific dataset for one strategy and run:

```bash
python3 -m embedding.visualization.spotlight_viewer \
  --strategy article_smart \
  --run-id <run_id>
```

The viewer resolves `spotlight_dataset.jsonl` from the embedding output layout
and requires a working Spotlight installation.
