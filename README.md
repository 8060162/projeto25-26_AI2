# Chunking project for regulatory PDFs

This project is a practical, modular starting point for chunking structured regulatory documents such as:

- regulations
- dispatches / legal notices
- annex-heavy institutional rules
- documents segmented by chapter, article, numbered items, and subitems

## High-level strategy

The project intentionally does **not** rely on one single naive splitter.
Instead, it supports three strategies:

1. **article_smart**
   - First split by article-like structure.
   - Then split long articles by numbered sections and subpoints.
   - Best first production strategy for your type of PDFs.

2. **structure_first**
   - Parse more structure first: preamble, chapter, article, numbered items.
   - Produce chunks from logical blocks.
   - Better when documents are consistently formatted.

3. **hybrid**
   - Try structure-first parsing.
   - Fall back to article-smart or paragraph-aware splitting when structure is weak.
   - Better for more variable documents.

## Recommended production flow

For your current corpus, I recommend this sequence:

### Phase 1
Use `article_smart` as the default strategy.

Why:
- easier to debug
- robust enough for regulations
- good balance between chunk cleanliness and implementation complexity

### Phase 2
Use `structure_first` on the same files and compare outputs.

Why:
- often produces more semantically precise chunks
- better metadata precision for chapter / article / section mapping

### Phase 3
Keep `hybrid` as the safety net when a document does not fully follow the expected structure.

## Cleaning approach

The project does **not** send raw PDF text directly into chunking.
A dedicated normalization phase runs first.

That phase removes or reduces:
- repeated headers
- repeated footers
- page counters like `3|14`
- institutional repeated banners
- line breaks caused by PDF layout extraction
- excessive whitespace
- duplicate artifacts around page transitions

The code is careful to preserve the meaningful legal content while minimizing layout noise.

## Output folders

The code expects:

- input PDFs in `/data/raw`
- outputs in `/data/chunks`

For each processed PDF, the pipeline writes:

- normalized text snapshots
- structured JSON
- chunk JSON
- inspection DOCX

The inspection DOCX is designed for human QA.
It helps you inspect:
- the text quality of each chunk
- metadata richness
- whether the semantic split makes sense

## Suggested first run

```bash
python main.py --strategy article_smart
```

Then inspect files under:

```bash
/data/chunks
```

## Suggested comparison workflow

Run the same corpus with the three strategies:

```bash
python main.py --strategy article_smart
python main.py --strategy structure_first
python main.py --strategy hybrid
```

Compare the generated DOCX files and JSON outputs.

## Main design principles used here

- clean text, rich metadata
- chunks should be semantically coherent
- chunk boundaries should not be blindly tied to article boundaries
- metadata should carry chapter / article / page context
- pipeline should be modular and easy to tune
- comments are intentionally verbose and in English for maintainability

## When you may want to provide more input later

You do **not** need to provide anything else to start.

However, later it may help to provide:
- 2 to 5 more PDFs with unusual formatting
- examples of bad chunks that you want to avoid
- your preferred maximum chunk size in tokens or characters
- whether annexes should be chunked differently from the main body

