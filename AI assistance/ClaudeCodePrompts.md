# Claude Code Prompt Log

Record of prompts given to Claude Code on this project, and what came of them.

---

## 001 — 2026-09-04 — HTML to JSON parsing

> We have an english version HTML of the EU AI Act. Goal is a RAG based Natural Language Interface in python to query an LLM about AI projects in the EU, answered using the EU AI Act as reference. Start by creating a JSON version of the EU AI Act using HTML parsing [+ target schema for `art_6.para_2`]. Each paragraph its own item in a list of items. Create code cells inside HTML_to_JSON.ipynb, preferably using BeautifulSoup.

**Result.** Wrote 9 code cells in `code/Data Processing/HTML_to_JSON.ipynb`. Ran end to end; produced `data/JSON/eu_ai_act.json` — a flat list of **849 items**, 0.7 MB.

| Type | Count | Coverage |
|---|---|---|
| `article` | 519 | Articles 1–113 |
| `recital` | 180 | Recitals 1–180 |
| `annex` | 150 | Annexes I–XIII |

Checks pass: no duplicate ids, no empty text, every article has a title.

**How it works.** EUR-Lex marks the act up with stable element ids — `art_6` for an
article, `006.002` for Article 6(2), `cpt_III.sct_1.tit_1` for a heading. Articles and
paragraphs come straight off those ids. Chapter and section are not stored on the
article, so they are recovered by walking the document in order and remembering the
most recent heading.

**Three things worth knowing about the source markup:**

1. **Enumerated points are tables, not lists.** Every `(a)`/`(b)` is a borderless
   table with the marker in a narrow first column, nested for sub-points.
   `extract_blocks` rebuilds them as indented text. Column count varies (2 in the
   Articles, 3 in Annex I), so the code takes the last cell as the body.
2. **Articles 107–109 quote other regulations they amend, reusing the quoted
   provision's id** — six ids appear twice in the file. Those quoted paragraphs are
   nested rather than direct children, so `recursive=False` keeps them inside their
   parent article rather than creating phantom duplicates of Article 5(4) and others.
3. **A `<p>` whose only child is an inline `<a>` produced no text** and was dropped
   silently. That cost three legislative acts in Annex X, so `extract_blocks` falls
   back to flat text when structural recursion yields nothing.

**Beyond the requested schema.** Recitals and Annexes are included alongside the
Articles, sharing the flat schema and distinguished by `type` (the field was already
in the spec). Annex III defines the high-risk categories and the recitals carry
interpretive intent, so a feasibility question cannot be answered from the Articles
alone.

---

## 002 — 2026-09-04 — Splitting Article 3 into definitions

> If Article 3 is presented in a single body, how do we split it up? I think its the one Article we should give special treatment to. Perhaps we could keep it under article 3 but change the type to "definitions" or smth?

**Result.** Agreed, and the numbers back it. Article 3 was 17,079 characters — nearly 4x the next largest single-body article (Article 66 at 4,507) and 28x the median. Split into one item per definition.

Corpus grew from 849 to **917 items**; longest single item fell from 17,079 to 4,701 characters.

| Type | Count |
|---|---|
| `article` | 519 |
| `definition` | 68 |
| `recital` | 180 |
| `annex` | 150 |

**Shape of a definition item** — `type` is `definition` (singular, to match `article` / `recital` / `annex`), and the defined term is lifted into its own `term` field, since a question like "what counts as a deployer" should match the term directly:

```json
{
  "id": "art_3.def_1",
  "type": "definition",
  "chapter": "I", "section": null,
  "article": 3, "paragraph": null,
  "definition": 1,
  "term": "AI system",
  "text": "'AI system' means a machine-based system that is designed to operate ...",
  "article_title": "Definitions"
}
```

All 68 definitions parsed, all 68 terms extracted. The lead-in sentence is kept as
`art_3.intro` so no text is lost. Definition numbers are read from the marker cell
rather than a loop counter.

**Also fixed:** a refactor had moved `item()` to module level with six parameters, but
the single-body branch still called `item(None, text)` with two. That raises
`TypeError` on the first unnumbered article — which is Article 3 — so the notebook
could not run to completion. Corrected while editing that cell.

Cells 15, 17 and 18 were updated too, since a new `type` value would otherwise have
made the "Articles 1-113 present" check report a false failure.

---

## 003 — 2026-09-04 — Combining Annex II into one item

> annex 2 has a structure that splits the points into single term text. Perhaps we can combine them all into a single JSON item instead?

**Result.** Agreed — Annex II was a clear outlier. Its points are bare terms with a
**median of 40 characters** and a minimum of 5 (`'rape,'`), against a median of 109+
for every other annex. As standalone chunks they retrieve as meaningless fragments,
and Article 5(1) refers to the list as one closed set.

Now emitted as a single 747-character item, `anx_II`, with `point: null` and the
intro line joined to all 16 offences. Corpus: 917 → **901 items** (annexes 150 → 134).

Implemented in cell 13 as a `WHOLE_ANNEXES = {"II"}` constant rather than a length
heuristic — an explicit set is predictable, and mirrors how Article 3 is special-cased.
Adding another annex is a one-word change.

**Left as-is, but flagged.** Annexes VIII (30 items, median 112) and XIII (8 items,
median 204) are also field lists with a few very short entries — `'the number of
registered end-users.'` at 35 characters. Nowhere near Annex II's outlier status, so
they were not touched; adding them to `WHOLE_ANNEXES` would combine them if retrieval
proves too fragmented.


---

## 004 — 2026-09-04 — Adding an embed_text field

> Can we create another field inside the JSON called embed_text? [...] The field should hold text that consists of the context header naming the provision [...] followed by the text of that paragraph, recital, annex point, definition, etc. This will allow us to perform similarity search on JSON list items based on their text and their header at the same time.

**Result.** Added `embed_text` to all 901 items. New notebook section **7. Text for
embedding** (two cells) defines `citation()` and `build_embed_text()`; the build cell
stamps the field onto every item before saving. Sections 7 and 8 renumbered to 8 and 9.

Header forms produced:

| Type | Header |
|---|---|
| article paragraph | `Article 6(2) — Classification rules for high-risk AI systems [Chapter III, Section 1]` |
| definition | `Article 3(1) — Definitions: 'AI system' [Chapter I]` |
| recital | `Recital 52 (AI Act preamble)` |
| annex point | `Annex III, point 4 — High-risk AI systems referred to in Article 6(2)` |
| whole annex | `Annex II — List of criminal offences referred to in Article 5(1)...` |

Header and text are joined by a newline, so `embed_text` always ends with the exact
`text` value — verified as an assertion in the checks cell.

**Why it earns its place.** The bare text of a provision is often anonymous. Article
6(2) reads "In addition to the high-risk AI systems referred to in paragraph 1..." and
never names Article 6, high-risk classification, or Chapter III. A query such as "which
article classifies high-risk systems" has almost nothing to match on without the header.

Definitions additionally carry the defined term in the header, since the term is what a
reader actually searches for.

Three new checks: every item has an `embed_text`, all 901 headers are **unique** (so a
header alone identifies a provision), and every `embed_text` ends with its own `text`.

JSON grew 0.7 → 1.3 MB, since `embed_text` restates `text` alongside the header.

---

## 005 — 2026-09-04 — Embedding stage and cache

> We will now focus on the actual NLI code (Natural_Language_Interface.ipynb). Load the JSON, embed the text within the new field, save embeddings to a numpy file, save embedding_ids matching embeddings to items, add embedding metadata. Use BAAI/bge-small-en-v1.5 with normalized embeddings. Run at the start of the application, compute once, and on later runs compare current metadata against saved metadata — rerun on id mismatch or model change, otherwise load from the numpy file.

**Result.** Built `code/Application/Natural_Language_Interface.ipynb`, 8 code cells,
runs clean cold and warm.

| Artefact | File | Detail |
|---|---|---|
| vectors | `data/Embeddings/eu_ai_act_embeddings.npy` | (901, 384) float32, 1.3 MB |
| row → id | `data/Embeddings/embedding_ids.json` | 901 ids in row order |
| metadata | `data/Embeddings/embedding_metadata.json` | model, fingerprint, versions, timings |

Cold run encoded 901 items in **24.2 s** on CPU; warm run loads from disk. All 384-dim
vectors are unit length to 1e-6.

**Cache design.** Six fields form the cache key: `model_name`, `embedded_field`,
`normalized`, `dtype`, `item_count`, `corpus_fingerprint` — the last a SHA-256 over
every `(id, embed_text)` pair in order, so an added, removed, reordered, renamed or
edited item all invalidate. Torch and sentence-transformers versions, device and
timestamps are recorded for traceability but deliberately do **not** force a rebuild,
since they do not change what the model produces. `cache_status()` returns the reasons
for a rebuild rather than a bare boolean, so a rebuild always explains itself. A
further check confirms the ids file and the vector matrix agree in length and order.

Invalidation was tested by tampering with each key in the saved metadata; all four
tested fields were detected, and the file restored.

**A real defect this surfaced.** `bge-small-en-v1.5` truncates at 512 tokens.
21 of 901 items (2.3%) exceed it, the worst being `art_5.para_1` at **950 tokens** —
the prohibited-practices list, arguably the most important provision in the act.
Only points (a) to (e) survive the cut; **(f), (g) and (h) are lost**: emotion
recognition, biometric categorisation, and real-time remote biometric identification.

This is not theoretical. The sanity query *"Is emotion recognition in the workplace
allowed?"* fails to retrieve Article 5(1) at all, because the sentence prohibiting it
was never embedded. Section 7 of the notebook reports the truncation count on every run
so it stays visible.

Not fixed in this prompt. The natural fix mirrors Article 3: split Article 5(1) into
one item per prohibited practice.

---

## 006 — 2026-09-04 — Fitting every item to the 512-token window

> OK can you perform these changes in the HTML_to_JSON notebook so we end up with JSON items that do not exceed 512 token limit?

**Result.** New section **8. Fitting the embedding model's token window** in
`HTML_to_JSON.ipynb`. Corpus 901 → **980 items**; items over the limit **21 → 0**,
longest now exactly 512 tokens.

| | before | after |
|---|---|---|
| items | 901 | 980 |
| over 512 tokens | 21 | **0** |
| longest item | 950 tok | 512 tok |

**Two strategies, applied in order.**

1. *Structural split* for provisions that are lists of lettered points (7 items).
   `art_5.para_1` becomes `art_5.para_1.point_a` … `point_h`, each 74–211 tokens, with
   the lead-in sentence carried onto every point so it reads standalone.
2. *Prose chunking* on sentence boundaries for solid prose with no points (14 items,
   mostly recitals), producing `rct_53.chunk_1`, `chunk_2`.

**Why structural rather than regex.** Splitting the flattened text on `(a)`/`(b)` looks
easy and is wrong: the corpus contains 59 inline cross-references of the form
"referred to in points (a) and (b)", and Article 5(1) interleaves roman sub-points
`(i)`/`(ii)` inside letter points. Splitting instead walks the source HTML tables, the
same approach `split_annex_points` already uses, so only genuine top-level points split.

**Citation updated.** `citation()` now emits `Article 5(1)(f) — Prohibited AI practices
[Chapter II]` for split points and `Recital 53 (AI Act preamble) [part 2]` for chunks.
All **980 headers remain unique**. The point is appended after the paragraph number,
not after the title — an ordering bug caught on the first run.

**Honest outcome on the motivating query.** The emotion-recognition prohibition was
previously *unreachable*: its text sat beyond the truncation point and was never
embedded. It is now indexed and retrievable — but for "Is emotion recognition in the
workplace allowed?" it ranks **6th**, behind recitals 44, 18 and 57, which discuss
emotion recognition in more natural language. On a keyword-style query it ranks 2nd.

So this prompt fixed what it set out to fix — nothing is truncated, and every provision
is now in the index. Getting the *operative prohibition* to outrank the *interpretive
recitals* is a retrieval-ranking problem for the next stage: a larger `k`, weighting
articles above recitals, or a reranker.
