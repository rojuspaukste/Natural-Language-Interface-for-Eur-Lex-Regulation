# Natural Language Interface for Eur-Lex Regulation

A retrieval-augmented question-answering prototype over **Regulation (EU) 2024/1689 — the EU AI Act**.

Ask a question in plain English; the system finds the provisions that actually answer it, hands only those to a language model, and returns an answer where every claim carries the Article, Annex or Recital it came from.

```
> Is a CV-screening system used for recruitment high-risk?

Yes. AI systems intended to be used for the recruitment or selection of natural
persons — in particular to place targeted job advertisements, analyse and filter
applications, and evaluate candidates — are listed in Annex III, point 4, and
systems referred to in Annex III are classified as high-risk (Article 6(2)).

This is subject to the derogation in Article 6(3): such a system is not high-risk
where it does not pose a significant risk of harm to health, safety or fundamental
rights, and does not materially influence the outcome of decision-making.

Sources: Article 6(2), Article 6(3), Annex III point 4
This is informational only and is not legal advice.
```

---

## Architecture

![Solution architecture](Documentation/architecture.png)

Two notebooks, run in order. The first builds the knowledge base; the second indexes it and answers questions.

---

## Repository layout

```
├── code/                      Notebooks and exported .py versions
│   ├── HTML_to_JSON.ipynb             1. Parse the regulation into a corpus
│   └── Natural_Language_Interface.ipynb   2. Embed, retrieve, answer
├── data/
│   ├── HTML/                  Source document as published by EUR-Lex
│   ├── JSON/                  Parsed corpus (eu_ai_act.json)
│   └── Embeddings/            Vectors, row ids, run metadata
├── Documentation/             Architecture diagram and design notes
├── AI assistance/             Prompts used and how the output was validated
└── .env                       GEMINI_API_KEY (git-ignored)
```

---

## 1. `HML_to_JSON.ipynb` — building the knowledge base

**Input:** the English HTML of the AI Act from EUR-Lex (CELEX `32024R1689`), saved locally.
**Output:** `data/JSON/eu_ai_act.json` — roughly 920 flat records.

| Step | What it does |
|---|---|
| Load the HTML | BeautifulSoup + lxml turn the document into a navigable tree |
| Detect provisions | Anchored regex on EUR-Lex ids: `art_6`, `006.002`, `rct_52`, `anx_III`, `cpt_III.sct_1` |
| Extract the content | Articles and their paragraphs, the 68 definitions of Article 3, 180 recitals, annex points, with chapter and section carried down from the surrounding structure |
| Add a citation header | Each record gets an `embed_text` field: a header naming the provision, then its text |
| Fit and validate | Split anything over the encoder's 512-token window; assert the corpus is complete |

### Why the structure matters

EUR-Lex publishes the act as ELI-annotated XHTML, where every provision has a stable, semantic id and the hierarchy is expressed as physical nesting. The parser follows that structure rather than splitting the text into fixed-size blocks, which gives three things a naive splitter cannot:

- **Verifiable citations.** Every record knows it is Article 6(2), so an answer can point at a provision and link back to EUR-Lex.
- **A meaningful distinction between binding and interpretive text.** Articles and Annexes are binding; Recitals are preamble. The `type` field carries this into the prompt.
- **Filtering for free.** Search only articles, only Annex III, only definitions — one condition each, no reindexing.

Two provisions get special treatment. **Article 3** holds all 68 definitions in a single 17,000-character body; left whole, a question about one defined term would retrieve all sixty-eight, so it is split into one record per definition with the defined term lifted into its own field. **Annex II** is the opposite case — a list of criminal offences as bare terms ("terrorism," "sabotage,") that mean nothing in isolation, so it is kept as one record.

### The 512-token window

`bge-small-en-v1.5` reads at most 512 tokens and **silently truncates** the rest — it does not error, it returns a confident vector describing only the opening. Twenty-one items exceeded the window. The worst was Article 5(1) at 950 tokens: points (a)–(e) survived, while **(f) emotion recognition, (g) biometric categorisation and (h) live facial recognition were never embedded at all.**

Two strategies fix it, applied in order:

1. **Structural split** — where a provision is a list of lettered points, split into one record per point, carrying the lead-in sentence so each stands alone. `art_5.para_1` becomes `art_5.para_1.point_a` … `point_h`, citing as *Article 5(1)(f)*. The split walks the source markup rather than pattern-matching flattened text, because markers like `(a)` also occur inside cross-references.
2. **Prose chunking** — recitals are solid prose with no points, so they are divided on sentence boundaries into the fewest chunks that fit.

### Record shape

```json
{
  "id": "art_6.para_2",
  "type": "article",
  "chapter": "III",
  "section": "1",
  "article": 6,
  "paragraph": 2,
  "text": "In addition to the high-risk AI systems referred to in paragraph 1, AI systems referred to in Annex III shall be considered to be high-risk.",
  "article_title": "Classification rules for high-risk AI systems",
  "embed_text": "Article 6(2) — Classification rules for high-risk AI systems [Chapter III, Section 1] In addition to the high-risk AI systems ..."
}
```

`text` is the authentic provision, used for display and citation. `embed_text` is what gets indexed. They are deliberately different: Article 6(2) never mentions Article 6, high-risk classification, or Chapter III, so on its own text alone it is close to unfindable — the header supplies the context the provision assumes.

---

## 2. `Natural_Language_Interface.ipynb` — indexing and answering

**Input:** `eu_ai_act.json`, plus a question typed by the user.
**Output:** vector index on disk, and an answer with citations.

### Indexing (once)

Every `embed_text` is encoded with **`BAAI/bge-small-en-v1.5`** into a 384-dimension unit vector, producing three artefacts in `data/Embeddings/`:

| File | Contents |
|---|---|
| `eu_ai_act_embeddings.npy` | `(N, 384)` float32 matrix; row *i* is corpus item *i* |
| `embedding_ids.json` | item ids in row order |
| `embedding_metadata.json` | model, dimension, normalisation, item count, SHA-256 fingerprint |

The row order is the only thing linking vectors to records, so the cache is invalidated by a **content fingerprint** over every `(id, embed_text)` pair, not just by file presence. Change the model, the field embedded, the item count, or a single character of any item's text, and the vectors are rebuilt. A silently misaligned index is the worst failure mode in a RAG system: it returns confident answers citing the wrong provision.

Embedding runs locally, so no document content leaves the machine at index time.

### Answering (per question)

| Step | What it does |
|---|---|
| Embed the question | Same model, with the BGE retrieval prefix, into one 384-number vector |
| Search | Cosine similarity against every item vector; keep the ten most similar |
| Assemble context | Expand each hit to the whole article or annex it belongs to, in document order, capped at 12,000 tokens |
| Build the prompt | Grounding rules plus the assembled provisions and the question |
| Generate | Gemini flash at temperature 0 |

**Why context assembly exists.** Retrieving the best-matching paragraph is not enough for legal text, because the rule and its exception live in different paragraphs. Ask *"are all Annex III systems high-risk?"* and Article 6(2) — short, punchy, a semantic bullseye — ranks first and says yes. Article 6(3), the derogation, says *no, not where certain conditions are met.* Returning the first without the second is a confidently wrong legal answer produced by textbook-perfect retrieval. So each hit is expanded to its whole provision before the model sees it.

Three rules keep expansion from exploding: definitions never expand (one hit would drag in all of Article 3), recitals expand only across their own chunks, and any group above the per-group token limit contributes only the items that actually matched. Groups are added best-scoring first, so anything dropped is the least relevant material.

**The prompt rules** are what keep the answer honest:

- answer only from the provisions supplied, with no outside knowledge
- Articles and Annexes are binding; Recitals are interpretive and never the source of a rule
- cite the provision immediately after each claim
- say so plainly when the provisions do not answer the question
- informational only, not legal advice

---

## Running it

```bash
pip install beautifulsoup4 lxml numpy torch sentence-transformers transformers google-genai python-dotenv
```

1. Put a Gemini API key in `.env` at the repository root: `GEMINI_API_KEY=...` (a free key from [AI Studio](https://aistudio.google.com/apikey) is sufficient). `.env` is git-ignored.
2. Run `code/HTML_to_JSON.ipynb` to build `data/JSON/eu_ai_act.json`.
3. Run `code/Natural_Language_Interface.ipynb`. The first run downloads the embedding model (~130 MB) and encodes the corpus; later runs load the cached vectors. The final cell opens a question loop — type `exit` to leave.

No GPU required; CPU encoding of the full corpus takes well under a minute.

---

## Validation

The parser is checked against facts about the regulation that are known independently of the code:

- 113 articles, 180 recitals, 13 annexes and 68 definitions all present
- no duplicate ids, no empty text, no article missing a title
- no line breaks, tabs, doubled spaces or untrimmed whitespace in any text field
- every `embed_text` begins with its citation header and ends with its text
- every item inside the encoder's 512-token window

The index is checked on every run: one vector per item, ids aligned with corpus order, no NaN or infinite values, all vectors unit length. Retrieval is spot-checked on the provisions a real question is most likely to need — Article 5, Article 6, Annex III, and the definitions.

---

## Limitations and next steps

- **Retrieval is dense-only.** Legal queries contain exact strings ("Annex III", "Article 53(1)(a)") that embeddings handle poorly. BM25 fused with the dense ranking via Reciprocal Rank Fusion is the first improvement.
- **No reranker.** Article 16 is *titled* "Obligations of providers of high-risk AI systems" and still ranks third for that exact question, behind two near-misses. A cross-encoder would fix the ordering.
- **Refusal is prompt-based only.** Out-of-scope questions score visibly lower than in-scope ones, so a score threshold would make refusal explicit rather than relying on the model's judgement.
- **English only.** All 24 language versions of an EU regulation are equally authentic. EUR-Lex ids are language-invariant — `art_6` is Article 6 in every version — so multilingual support is a metadata dimension rather than 24 parallel pipelines.
- **The corpus is static.** Regulations are amended and corrected. Production would re-ingest on a schedule by CELEX id and keep corpus versions.
- **No evaluation numbers yet.** A gold set of questions with known-correct provision ids, scored by recall@k and unsupported-citation rate, is the missing measurement.

---

## AI assistance

Coding assistants were used to generate mechanical code from specifications decided beforehand. The prompts used, and how each output was validated, are documented in [`AI assistance/`](AI%20assistance/).

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built as a take-home exercise. Not legal advice; the authentic text of the regulation is the one published on [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng).*
