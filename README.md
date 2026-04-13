# Regulatory PDF Chunking, Embedding, and Retrieval Pipeline

This project processes structured regulatory PDFs through three connected phases:

1. chunking, which extracts structure, normalizes text, parses the document tree, and exports chunk artifacts
2. embedding, which loads the chunk output for the configured strategy and generates vectors through a provider-based embedding step
3. retrieval, which embeds a user question, queries ChromaDB Cloud, builds grounded context, applies deterministic guardrails, and generates a response

The retrieval phase now supports a full `question -> retrieval -> response` flow. Query embedding reuses the configured embedding stack, vector search runs against ChromaDB Cloud, and the answer step is isolated behind a response-generation adapter.

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
  },
  "retrieval": {
    "enabled": true,
    "top_k": 8,
    "score_filtering": {
      "enabled": false,
      "min_similarity_score": 0.0
    },
    "context": {
      "max_chunks": 4,
      "max_characters": 12000
    }
  },
  "response_generation": {
    "enabled": true,
    "provider": "openai",
    "model": "gpt-4o",
    "grounded_fallback_enabled": true
  },
  "guardrails": {
    "enabled": true,
    "pre_request": {
      "offensive_language_checks_enabled": true,
      "sexual_content_checks_enabled": true,
      "discriminatory_content_checks_enabled": true,
      "criminal_or_dangerous_content_checks_enabled": true,
      "sensitive_data_checks_enabled": true,
      "dangerous_command_checks_enabled": true
    },
    "post_response": {
      "unsafe_output_checks_enabled": true,
      "grounded_response_checks_enabled": true,
      "unsupported_answer_checks_enabled": true
    }
  },
  "metrics": {
    "enabled": true,
    "track_deflection_rate": true,
    "track_false_positive_rate": true,
    "track_jailbreak_resistance": true,
    "track_stage_latency": true
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
- retrieval execution is enabled or disabled through `retrieval.enabled`
- retrieval queries ChromaDB through the existing storage layer and uses the same configured embedding provider/model family for question embedding
- response generation is configured under `response_generation`
- deterministic guardrails are configured under `guardrails`
- retrieval metrics are configured under `metrics`

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

### Retrieval phase

The retrieval pipeline runs after embeddings already exist in ChromaDB Cloud.

It:

- embeds the user question with the configured embedding provider
- queries the configured ChromaDB collection through the storage layer
- builds compact grounded context from the retrieved chunks
- generates the final answer through the configured response-generation adapter
- applies deterministic guardrails before retrieval and after answer generation
- records retrieval metrics for safety and latency analysis

Current deterministic guardrails include:

- offensive language checks
- sexual content checks
- discriminatory content checks
- criminal or dangerous content checks
- sensitive data checks
- dangerous command checks
- unsafe output checks
- grounded-response checks
- unsupported-answer checks

Current tracked retrieval metrics include:

- `Deflection Rate`
- `False Positive Rate`
- `Jailbreak Resistance`
- stage and total latency overhead

Retrieval is executed with:

```bash
python3 retrieval/main.py --question "Qual e o prazo de inscricao?"
```

Before running retrieval:

- set `retrieval.enabled` to `true`
- confirm the configured `embedding.provider`, `embedding.model`, and `embedding.chromadb.*` values match the stored vectors
- ensure embeddings already exist in the configured ChromaDB Cloud collection
- set `response_generation.enabled` to `true`
- export `CHROMA_API_KEY` when `embedding.chromadb.mode` is `cloud`
- export `OPENAI_API_KEY` when `response_generation.provider` is `openai`

Important note:

- this release implements deterministic rule-based guardrails as the first safety baseline
- model-based guardrails are not part of the current implementation and should be treated as a future extension only

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
7. Run `python3 retrieval/main.py --question "<your question>"`.

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
- retrieval backed by ChromaDB Cloud through the existing storage layer
- grounded question answering built from retrieved chunk context
- deterministic guardrails enforced before and after answer generation
- lightweight metrics for safety, robustness, and latency
- Spotlight kept as a supported inspection layer
- outputs suitable for manual QA and downstream retrieval workflows

