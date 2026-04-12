# Regulatory PDF Chunking and Embedding Pipeline

This project processes structured regulatory PDFs through two separate phases:

1. chunking, which extracts structure, normalizes text, parses the document tree, and exports chunk artifacts
2. embedding, which loads the chunk output for the configured strategy and generates vectors through a provider-based embedding step

The downstream retrieval flow is intended for a final OpenAI GPT-4o agent via API. Vector generation is handled separately by the embedding provider and is not the same step as the final GPT-4o agent call.

## Supported document types

The pipeline is designed for documents such as:

- regulations
- dispatches and legal notices
- annex-heavy institutional rules
- PDFs segmented by chapter, article, numbered items, and subitems

## Central configuration

Runtime behavior is controlled from `config/appsettings.json`.

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

Important points:

- `chunking.strategy` is now the single source of truth for chunking strategy selection
- chunking no longer depends on `--strategy` in the normal execution flow
- embedding reads the same configured strategy and only loads chunk outputs generated for that strategy
- embedding execution is enabled or disabled through `embedding.enabled`
- embedding provider and model are configured independently from the final GPT-4o agent
- the main embedding flow uses `sentence_transformers` with `all-MiniLM-L6-v2`
- ChromaDB settings are configured under `embedding.chromadb`

## Chunking strategies

The project supports three chunking strategies:

1. `article_smart`
2. `structure_first`
3. `hybrid`

The active one is selected in `config/appsettings.json`, not at runtime through a CLI strategy flag.

## Pipeline overview

### Chunking phase

The chunking pipeline:

- reads PDFs from `data/raw`
- extracts document text and structure
- runs extraction quality analysis
- applies OCR fallback when needed
- normalizes PDF noise conservatively
- parses the document into a structural tree
- exports structure artifacts and chunk artifacts for the configured strategy

Chunking is executed with:

```bash
python3 src/main.py
```

### Embedding phase

The embedding pipeline is a separate step that runs only after chunk outputs already exist.

It:

- reads the configured strategy from `config/appsettings.json`
- discovers `05_chunks.json` files under the configured input root
- builds the text sent for embedding
- generates vectors through the configured embedding provider
- uses `sentence-transformers/all-MiniLM-L6-v2` in the default configured flow
- persists embeddings to ChromaDB as the main storage
- writes local auxiliary artifacts for audit and visualization support
- optionally exports a Renumics Spotlight dataset

Embedding is executed with:

```bash
python3 -m embedding.main
```

Before running embedding:

- set `embedding.enabled` to `true`
- ensure chunk outputs already exist for the configured strategy
- confirm the configured `embedding.provider`, `embedding.model`, and `embedding.chromadb.*` values
- export `CHROMA_API_KEY` only when `embedding.chromadb.mode` is `cloud`
- export `OPENAI_API_KEY` only when `embedding.provider` is explicitly switched to `openai`

## Output folders

Expected input:

- PDFs in `data/raw`

Chunking outputs:

- structure artifacts under `data/chunks/<doc_id>/structure/`
- strategy-specific chunk artifacts under `data/chunks/<doc_id>/<strategy>/`

Embedding outputs:

- the latest run artifacts under `data/embeddings/<strategy>/<run_id>/`
- each new embedding execution removes the previous persisted local run for that strategy before writing the new one
- ChromaDB remains the source of truth for stored embedding vectors

Typical embedding artifacts:

- `chromadb_storage.json`
- `run_manifest.json`
- `spotlight_dataset.jsonl` when Spotlight export is enabled

## Recommended working flow

1. Choose the active strategy in `config/appsettings.json`.
2. Run `python3 src/main.py`.
3. Inspect the generated JSON and DOCX artifacts under `data/chunks`.
4. Enable embedding in `config/appsettings.json` when chunk quality is acceptable.
5. Run `python3 -m embedding.main`.
6. Inspect ChromaDB plus the local auxiliary artifacts under `data/embeddings`.

## Visualization

Renumics Spotlight is supported as a visualization and inspection tool for embeddings.

To export Spotlight data during embedding execution, enable:

```json
"visualization": {
  "enabled": true,
  "spotlight_enabled": true
}
```

Then run:

```bash
python3 -m embedding.main
python3 -m embedding.visualization.spotlight_viewer
```

## Design principles

- clean text and traceable metadata
- settings-driven execution
- chunking and embedding kept as separate phases
- minimal PDF noise carried into chunks and embeddings
- provider-based embedding generation
- ChromaDB as the main embedding storage with local auxiliary artifacts where needed
- Spotlight kept as a supported inspection layer
- outputs suitable for manual QA and downstream retrieval workflows

