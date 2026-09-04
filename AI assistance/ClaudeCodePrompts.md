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
