# Regulatory PDF Chunking, Embedding, and Retrieval Pipeline

This project processes structured regulatory PDFs through three connected phases:

1. chunking, which extracts structure, normalizes text, parses the document tree, and exports chunk artifacts
2. embedding, which loads the chunk output for the configured strategy and generates vectors through a provider-based embedding step
3. retrieval, which embeds a user question, queries ChromaDB Cloud, builds grounded context, applies deterministic guardrails, and generates a response

The retrieval phase now supports a full `question -> retrieval -> response` flow. Query embedding reuses the configured embedding stack, vector search runs against ChromaDB Cloud, and the answer step is isolated behind a response-generation adapter.

## Current fixing cycle

The current retrieval and guardrails fixing cycle is focused on the remaining runtime-quality problems in legal question answering, not on rebuilding the stack from scratch.

This cycle documents and preserves behavior that is already implemented in the repository:

- richer legal-intent normalization for PT-PT legal questions
- dynamic document inference when the regulation title is omitted
- primary-anchor-based context selection with graded legal competitors
- first-pass vs second-pass retrieval recovery
- cautious-answer vs deflection behavior for under-specified legal questions
- grounding validation for citations and conservatively supported derived claims
- benchmark-driven protection for semantic-equivalence and vague-question failures

This cycle is not focused on changing the embedding model or reworking answer-provider integration. The `external_gpt4o` adapter already exists, remains supported, and is documented here only as an operational runtime option.

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
    "comparison": {
      "enabled": false,
      "candidate_models": [
        "all-MiniLM-L6-v2",
        "Qwen/Qwen3-VL-Embedding-8B"
      ],
      "output_root": "data/embedding_comparison"
    },
    "chromadb": {
      "mode": "cloud",
      "persist_directory": "data/chromadb",
      "collection_name": "rag_embeddings",
      "cloud": {
        "tenant": "",
        "database": "",
        "host": "",
        "port": 443,
        "api_key_env_var": "CHROMA_API_KEY"
      }
    },
    "visualization": {
      "enabled": true,
      "spotlight_enabled": true,
      "benchmark_overlay": {
        "enabled": false,
        "output_path": "data/embeddings/benchmark_overlay.jsonl"
      }
    }
  },
  "retrieval": {
    "enabled": true,
    "top_k": 8,
    "candidate_pool_size": 12,
    "query_normalization": {
      "enabled": true,
      "strip_formatting_instructions": true,
      "extract_formatting_directives": true
    },
    "score_filtering": {
      "enabled": false,
      "min_similarity_score": 0.0
    },
    "context": {
      "max_chunks": 4,
      "max_characters": 12000,
      "include_article_number": true,
      "include_article_title": true,
      "include_parent_structure": true,
      "primary_anchor_max_count": 1,
      "single_intent_compaction_enabled": true,
      "single_intent_max_supporting_chunks": 1
    },
    "routing": {
      "enabled": true,
      "article_scoping_enabled": true,
      "document_scoping_enabled": true,
      "comparative_retrieval_enabled": true,
      "weak_evidence_retry_enabled": true,
      "document_inference_enabled": true,
      "document_inference_min_score": 3.0,
      "document_inference_min_margin": 2.0,
      "scoped_candidate_pool_size": 8,
      "broad_candidate_pool_size": 16
    },
    "second_pass_retry": {
      "enabled": true,
      "candidate_pool_size": 32,
      "dominant_document_min_share": 0.6
    },
    "evidence_quality": {
      "strong_min_score": 0.75,
      "weak_min_score": 0.45,
      "ambiguity_score_margin": 0.05,
      "conflict_score_margin": 0.1,
      "non_blocking_competitor_score_margin": 0.08,
      "blocking_conflict_score_margin": 0.15
    },
    "grounding": {
      "derived_claims_enabled": true,
      "derived_numeric_claim_tolerance": 0,
      "enumeration_coverage_threshold": 0.8
    }
  },
  "response_generation": {
    "enabled": true,
    "provider": "external_gpt4o",
    "model": "gpt-4o",
    "grounded_fallback_enabled": true,
    "openai": {
      "api_key_env_var": "OPENAI_API_KEY"
    },
    "external_gpt4o": {
      "endpoint_url": "",
      "auth_env_var": "EXTERNAL_GPT4O_API_KEY",
      "question_field_name": "message",
      "context_field_name": "context",
      "instructions_field_name": "instructions",
      "metadata_field_name": "metadata",
      "channel_id": "",
      "thread_id": "",
      "user_info": "{}",
      "auth_header_name": "x-api-key",
      "auth_header_prefix": "",
      "timeout_seconds": 30,
      "max_retries": 2
    }
  },
  "guardrails": {
    "enabled": true,
    "portuguese_coverage": {
      "enabled": true,
      "jailbreak_pattern_checks_enabled": true
    },
    "pre_request": {
      "offensive_language_checks_enabled": true,
      "sexual_content_checks_enabled": true,
      "discriminatory_content_checks_enabled": true,
      "criminal_or_dangerous_content_checks_enabled": true,
      "sensitive_data_checks_enabled": true,
      "dangerous_command_checks_enabled": true,
      "jailbreak_pattern_checks_enabled": true
    },
    "post_response": {
      "unsafe_output_checks_enabled": true,
      "grounded_response_checks_enabled": true,
      "unsupported_answer_checks_enabled": true,
      "sensitive_data_checks_enabled": true
    }
  },
  "metrics": {
    "enabled": true,
    "track_deflection_rate": true,
    "track_false_positive_rate": true,
    "track_jailbreak_resistance": true,
    "track_stage_latency": true,
    "retrieval_quality": {
      "enabled": true,
      "track_candidate_pool_size": true,
      "track_selected_context_size": true,
      "track_context_truncation": true,
      "track_structural_richness": true
    }
  },
  "benchmark": {
    "enabled": false,
    "datasets": {
      "questions_path": "benchmark/questions.jsonl",
      "guardrails_path": "benchmark/guardrails.jsonl"
    },
    "outputs": {
      "output_root": "data/benchmark_runs"
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
- candidate embedding models for controlled comparison are configured under `embedding.comparison`
- ChromaDB settings are configured under `embedding.chromadb`
- Spotlight and benchmark-question overlay exports are configured under `embedding.visualization`
- retrieval execution is enabled or disabled through `retrieval.enabled`
- retrieval queries ChromaDB through the existing storage layer and uses the same configured embedding provider/model family for question embedding
- retrieval can normalize the raw question into a cleaner semantic query through `retrieval.query_normalization.*` while preserving the original wording for answer generation
- retrieval routing is configured under `retrieval.routing` and controls broad, expanded-broad, scoped, comparative, article-biased, and retry-candidate retrieval behavior
- dynamic document inference is configured under `retrieval.routing.*` and uses generic query and evidence signals instead of document-specific runtime hardcodes
- first-pass and second-pass retry behavior is configured under `retrieval.second_pass_retry`
- retrieval asks storage for the broader configured candidate pool and lets the context builder decide what survives into the final answer context
- primary-anchor selection, single-intent compaction, and structural grounding fields exposed in the packed context are configured under `retrieval.context.*`
- evidence strength, ambiguity, close competitors, and blocking-conflict thresholds are configured under `retrieval.evidence_quality`
- derived-claim grounding tolerance is configured under `retrieval.grounding`
- response generation is configured under `response_generation`
- response generation supports `external_gpt4o` for the configured academic endpoint and `openai` / `openai_direct` for the official OpenAI API
- the external GPT-4o answer adapter is configured under `response_generation.external_gpt4o` and reads its secret from an environment variable
- the official OpenAI adapter reads its secret from `response_generation.openai.api_key_env_var`, which defaults to `OPENAI_API_KEY`
- deterministic guardrails, including stronger Portuguese coverage and jailbreak checks, are configured under `guardrails`
- retrieval metrics, including optional retrieval-quality signals, are configured under `metrics`
- benchmark datasets and output roots are configured under `benchmark`

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

- preserves the original user question for downstream answer generation
- optionally normalizes the raw wording into a cleaner semantic retrieval query and extracts legal-intent signals such as article references, document titles, comparative intent, and likely regulatory subject matter
- routes the normalized query before vector search when retrieval routing is enabled
- uses generic legal-intent metadata and dynamic document inference when a question omits the regulation title
- embeds that semantic retrieval query with the configured embedding provider
- queries the configured ChromaDB collection through the storage layer using the route-selected candidate-pool breadth
- lets the context builder rank, deduplicate, filter, and trim that broader pool before final context packing
- builds compact grounded context from the retrieved chunks around a primary evidence anchor when the selected evidence supports one
- preserves structural legal anchors such as document title, article number, article title, and parent structure in the final context when configured
- classifies close legal competitors as supportive, alternative scope, or blocking conflict before deciding whether evidence is usable
- can run a second retrieval pass for broad or expanded-broad questions when first-pass evidence is weak, conflicting, or recoverable through an inferred document target
- generates the final answer through the configured response-generation adapter
- validates answer grounding, citation alignment, and supported derived claims after generation
- applies deterministic guardrails before retrieval and after answer generation
- records retrieval metrics for safety, latency, and context-quality analysis

The retrieval flow now separates two question representations:

- `question_text`: the original wording kept for answer generation and user-facing behavior
- `normalized_query_text`: the semantic retrieval query used for embedding and vector search when query normalization is enabled

This separation keeps formatting directives such as output language or citation style out of the vector search step while preserving them for the final answer.

Retrieval routing in this project is deterministic and evidence-oriented. It decides whether the question should use broad retrieval, expanded-broad retrieval, document-scoped retrieval, article-biased retrieval, retry-candidate document-scoped retrieval, or comparative multi-document retrieval. When the document is not explicitly named, the router can infer a likely document from generic legal-intent signals and active chunk metadata; ambiguous questions remain broad instead of being forced into a brittle document choice. Routing metadata also carries candidate-pool and retry decisions so the service can continue with normal generation, run second-pass recovery, produce a cautious answer, or deflect when evidence remains weak, ambiguous, or conflicting.

Context building centers single-intent legal questions on a primary evidence anchor when the retrieved candidates support one. Nearby legal chunks are kept as supporting evidence, alternative-scope competitors, or blocking conflicts according to deterministic evidence thresholds. This keeps final context compact without hiding close legal competitors that downstream validation or metrics need to inspect.

Second-pass retry is service-level orchestration. If the first pass uses a broad profile and produces weak, conflicting, or otherwise recoverable evidence, the service can infer a dominant document from first-pass candidates or selected anchors, retrieve a broader document-focused candidate pool, rebuild context, and validate the improved evidence before deciding whether to answer or deflect.

Grounding validation is a post-generation check. The validator compares the generated answer with the selected retrieval context, verifies cited document and article anchors when present, and rejects unsupported legal claims such as numeric deadlines or amounts that are not supported by the selected chunks. It can accept conservative derived claims, such as compact summaries of enumerated payment schedules, when those claims are clearly supported by the selected primary evidence and citations remain aligned.

Current deterministic guardrails include:

- offensive language checks
- sexual content checks
- discriminatory content checks
- criminal or dangerous content checks
- sensitive data checks
- dangerous command checks
- jailbreak-pattern checks when enabled
- unsafe output checks
- sensitive-data exposure checks in generated output
- grounded-response checks
- unsupported-answer checks
- Portuguese-aware coverage for pre-request categories, jailbreak attempts, sensitive data, destructive intents, and unsafe or unsupported output patterns

Current tracked retrieval metrics include:

- `Deflection Rate`
- `False Positive Rate`
- `Jailbreak Resistance`
- stage and total latency overhead
- requested retrieval breadth vs returned retrieval breadth
- candidate pool size vs selected context size
- context truncation
- structural richness of the selected context
- retry triggered count and retry success count
- first-pass and second-pass evidence classifications
- primary-anchor presence and stability
- deflection reason categories for evidence routing, grounding validation, pre-guardrails, and post-guardrails
- optional labeled relevant-chunk recovery counters

Retrieval is executed with:

```bash
python3 retrieval/main.py --question "Qual e o prazo de inscricao?"
```

Benchmark evaluation is executed with:

```bash
python3 -m retrieval.evaluation.main --mode full --run-id latest
```

Supported benchmark modes are `retrieval`, `answer`, `guardrails`, and `full`. The default datasets are `benchmark/questions.jsonl` and `benchmark/guardrails.jsonl`; the runner writes summaries under `data/benchmark_runs/<run_id>/`. Use `--questions-path`, `--guardrails-path`, `--retrieval-observations`, or `--answer-observations` for controlled deterministic inputs.

The question benchmark includes semantic-equivalence cases for PT-PT legal questions. These cases compare explicit-document and implicit-document formulations that should resolve to the same governing evidence, such as international-student payment-plan questions that mention the regulation title in one variant and omit it in another. Benchmark labels include expected route metadata, document ids, article numbers, required facts, forbidden facts, and grounding labels so retrieval quality, answer quality, and citation alignment can be evaluated together.

Before running retrieval:

- set `retrieval.enabled` to `true`
- confirm the configured `embedding.provider`, `embedding.model`, and `embedding.chromadb.*` values match the stored vectors
- ensure embeddings already exist in the configured ChromaDB Cloud collection
- review `retrieval.query_normalization.*`, `retrieval.routing.*`, `retrieval.second_pass_retry.*`, `retrieval.evidence_quality.*`, `retrieval.grounding.*`, `retrieval.top_k`, `retrieval.candidate_pool_size`, and `retrieval.context.*` so routing, retry behavior, retrieval breadth, final context packing, and structural grounding match the intended behavior
- set `response_generation.enabled` to `true`
- export `CHROMA_API_KEY` when `embedding.chromadb.mode` is `cloud`
- use `"response_generation.provider": "external_gpt4o"` for the configured academic GPT-4o endpoint and export the environment variable named by `response_generation.external_gpt4o.auth_env_var`
- use `"response_generation.provider": "openai"` or `"openai_direct"` for the official OpenAI API and export the environment variable named by `response_generation.openai.api_key_env_var`
- when `response_generation.provider` is `external_gpt4o`, set `response_generation.external_gpt4o.endpoint_url`, configure `channel_id`, `thread_id`, and `user_info` when the endpoint requires those operational fields, and export the environment variable named by `response_generation.external_gpt4o.auth_env_var`
- enable `guardrails.portuguese_coverage.*` and pre-request jailbreak checks when Portuguese deterministic safety coverage is required
- enable `metrics.retrieval_quality.*` when retrieval/context quality signals should be collected explicitly

Important note:

- this release implements deterministic rule-based guardrails as the first safety baseline
- model-based guardrails are not part of the current implementation and should be treated as a future extension only
- the current fixing cycle is centered on retrieval quality, grounding quality, benchmark coverage, and deterministic guardrails rather than on answer-provider changes
- the external GPT-4o adapter sends grounded generation requests through a configurable multipart HTTP endpoint; secrets are never stored in `config/appsettings.json`
- answer-generation rate limiting is logged as a warning for both the external GPT-4o endpoint and the official OpenAI API, including the configured provider and environment-variable name but never the secret value

### Asking questions with either answer provider

Use the same retrieval command in both modes:

```bash
python3 -m retrieval.main --question "Em que condições pode ser pedido um plano específico de pagamento de propina?"
```

For the academic external GPT-4o endpoint, keep `response_generation.provider` as `external_gpt4o` in `config/appsettings.json`, confirm the endpoint fields under `response_generation.external_gpt4o`, and export the configured secret:

```bash
export EXTERNAL_GPT4O_API_KEY="..."
python3 -m retrieval.main --question "Qual é o prazo aplicável?"
```

For the official OpenAI API, switch only the response-generation provider and ensure the key env var exists:

```json
"response_generation": {
  "enabled": true,
  "provider": "openai",
  "model": "gpt-4o",
  "openai": {
    "api_key_env_var": "OPENAI_API_KEY"
  }
}
```

```bash
export OPENAI_API_KEY="..."
python3 -m retrieval.main --question "Qual é o prazo aplicável?"
```

The `openai_direct` provider alias is also accepted and resolves to the same official OpenAI adapter.

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

Benchmark and evaluation outputs:

- benchmark summaries under `data/benchmark_runs/<run_id>/`
- embedding comparison summaries under `data/embedding_comparison/<run_id>/`
- benchmark-question overlay datasets at `data/embeddings/benchmark_overlay.jsonl` unless overridden

## Recommended working flow

1. Choose the active strategy in `config/appsettings.json`.
2. Run `python3 src/main.py`.
3. Inspect the generated JSON and DOCX artifacts under `data/chunks`.
4. Enable embedding in `config/appsettings.json` when chunk quality is acceptable.
5. Run `python3 -m embedding.main`.
6. Inspect ChromaDB plus the local auxiliary artifacts under `data/embeddings`.
7. Run `python3 retrieval/main.py --question "<your question>"`.
8. Run `python3 -m retrieval.evaluation.main --mode full --run-id latest`.
9. Compare candidate embedding models only through `python3 -m embedding.evaluation.embedding_benchmark`.

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

To export benchmark questions into the same embedding-space dataset as chunk vectors, first generate `spotlight_dataset.jsonl` through the embedding flow, then run:

```bash
python3 -m embedding.visualization.benchmark_overlay_main \
  --strategy article_smart \
  --run-id <run_id>
```

The overlay exporter reads chunk vectors from a Spotlight JSONL export, embeds benchmark questions with the configured embedding provider, and writes a combined JSONL dataset for inspection.

## Embedding comparison

Embedding-model changes are measured through the benchmark flow before any production model change is recommended. The active configured model is used as the baseline, and `Qwen/Qwen3-VL-Embedding-8B` is always included as a candidate unless it is already present.

Run the comparison with:

```bash
python3 -m embedding.evaluation.embedding_benchmark --run-id latest
```

Optional arguments include `--questions-path`, `--output-root`, `--model`, `--top-k`, and `--no-write`. The comparison writes `embedding_comparison_summary.json` under `data/embedding_comparison/<run_id>/` when artifact writing is enabled.

## Design principles

- clean text and traceable metadata
- settings-driven execution
- chunking and embedding kept as separate phases
- minimal PDF noise carried into chunks and embeddings
- provider-based embedding generation
- ChromaDB as the main embedding storage with local auxiliary artifacts where needed
- retrieval backed by ChromaDB Cloud through the existing storage layer
- deterministic retrieval routing kept separate from vector-store access
- dynamic document inference driven by generic legal-intent and evidence signals
- primary-anchor context selection kept in the context builder
- second-pass recovery orchestrated in the retrieval service
- grounded question answering built from retrieved chunk context
- grounding, citation validation, and supported derived-claim validation kept separate from answer generation
- deterministic guardrails enforced before and after answer generation
- benchmark datasets used as the source of truth for retrieval, answer, guardrail, and semantic-equivalence quality
- lightweight metrics for safety, robustness, and latency
- Spotlight kept as a supported inspection layer
- benchmark-question overlays kept as auxiliary evaluation artifacts, not runtime persistence
- outputs suitable for manual QA and downstream retrieval workflows
