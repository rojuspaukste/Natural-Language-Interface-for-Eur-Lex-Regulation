# %% [markdown]
# # EU AI Act — Natural Language Interface
# 
# A retrieval-augmented interface for asking questions about AI projects in the EU,
# answered from Regulation (EU) 2024/1689.
# 
# This notebook covers the **embedding stage**: load the parsed corpus, embed each
# item's `embed_text`, and cache the result so the cost is paid once.
# 
# | Artefact | File |
# |---|---|
# | corpus | `data/JSON/eu_ai_act.json` |
# | vectors | `data/Embeddings/eu_ai_act_embeddings.npy` |
# | row → item id | `data/Embeddings/embedding_ids.json` |
# | run metadata | `data/Embeddings/embedding_metadata.json` |
# 
# **Caching.** Every run builds a metadata record describing what *would* be produced,
# including a SHA-256 fingerprint over every `(id, embed_text)` pair. If that record
# matches the one saved beside the vectors, the vectors are loaded from disk. If
# anything meaningful differs — the model, the field embedded, normalisation, the item
# count, or a single character of any item's text — the embeddings are rebuilt.
# 

# %% [markdown]
# ## 1. Setup
# 
# Configuration lives in one place so a change to any of it invalidates the cache.
# 

# %%
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import sentence_transformers
from sentence_transformers import SentenceTransformer
from collections import OrderedDict, defaultdict

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
LLM_MODEL = "gemini-3.5-flash"     # check AI Studio for what your free tier allows

def find_project_root(start: Path) -> Path:
    """Find the repo root by looking for data/JSON, so this runs from anywhere."""
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "JSON").is_dir():
            return candidate
    raise FileNotFoundError(f"No data/JSON directory found above {start}")


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
JSON_PATH = PROJECT_ROOT / "data" / "JSON" / "eu_ai_act.json"
EMBED_DIR = PROJECT_ROOT / "data" / "Embeddings"
VECTORS_PATH = EMBED_DIR / "eu_ai_act_embeddings.npy"
IDS_PATH = EMBED_DIR / "embedding_ids.json"
METADATA_PATH = EMBED_DIR / "embedding_metadata.json"

# --- Embedding configuration -------------------------------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_FIELD = "embed_text"      # the field whose text becomes a vector
NORMALIZE = True                # unit vectors, so cosine similarity is a dot product
DTYPE = "float32"
BATCH_SIZE = 32

# BGE models expect short queries to carry a retrieval instruction, while the
# passages themselves are embedded bare. Used at query time, not here.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("project root :", PROJECT_ROOT)
print("model        :", MODEL_NAME)
print("device       :", DEVICE)
print("torch        :", torch.__version__)
print("sentence-tf  :", sentence_transformers.__version__)


# %% [markdown]
# ### LLM client
# 
# The answer-generation model is Google Gemini. The API key is read from a `.env` file at
# the repository root, which `.gitignore` excludes — the key must never reach the public
# GitHub remote. `.env.example` is committed as a template.
# 
# Setup, once:
# 
# 1. Open `.env` in the repository root.
# 2. Paste your key after `GEMINI_API_KEY=` (no quotes, no spaces).
# 3. Restart the kernel, then run this cell.
# 
# Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

# %%
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Explicit path: the notebook's working directory is code/Application, while
# .env lives at the repository root.
load_dotenv(PROJECT_ROOT / ".env")

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is empty. Paste your key into the .env file at "
        f"{PROJECT_ROOT / '.env'}, then restart the kernel and re-run."
    )

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
LLM_MODEL = "gemini-3.5-flash"     # confirm against the list below

print("Gemini client ready, model:", LLM_MODEL)

# %% [markdown]
# Run this once to see which models your key can actually reach, then set
# `LLM_MODEL` above to whichever you want. Free-tier availability changes, so trust this
# list over any hardcoded name.

# %%
# Models this API key can call
for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or []
    if "generateContent" in actions:
        print(" ", model.name)

# %% [markdown]
# ## 2. Embedding the JSON
# 

# %% [markdown]
# ### Loading the corpus

# %%
corpus = json.loads(JSON_PATH.read_text(encoding="utf-8"))

missing = [i["id"] for i in corpus if not i.get(EMBED_FIELD, "").strip()]
if missing:
    raise ValueError(f"{len(missing)} items have no {EMBED_FIELD}, e.g. {missing[:5]}")

print(f"{len(corpus)} items loaded from {JSON_PATH.name}")
for kind in ("article", "definition", "recital", "annex"):
    print(f"  {kind:11} {sum(1 for i in corpus if i['type'] == kind):4}")


# %% [markdown]
# ### Describing the run
# 
# The fingerprint is the heart of the cache. It hashes every item's id **and** its
# embedding text in order, so it changes if an item is added, removed, reordered,
# renamed, or edited — any of which would leave the saved vectors misaligned with the
# corpus.
# 
# `CACHE_KEYS` names the fields that must match for cached vectors to be reusable.
# Everything else in the metadata is recorded for traceability but does not force a
# rebuild: a newer torch does not change what this model produces.
# 

# %%
# Fields that must match for the cached vectors to be considered valid.
CACHE_KEYS = ("model_name", "embedded_field", "normalized", "dtype",
              "item_count", "corpus_fingerprint")


def corpus_fingerprint(items: list[dict]) -> str:
    """SHA-256 over every (id, embed_text) pair, in order."""
    digest = hashlib.sha256()
    for item in items:
        digest.update(item["id"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(item[EMBED_FIELD].encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def build_run_metadata(items: list[dict]) -> dict:
    """Describe what this run would produce, before producing it."""
    return {
        # --- cache keys ---
        "model_name": MODEL_NAME,
        "embedded_field": EMBED_FIELD,
        "normalized": NORMALIZE,
        "dtype": DTYPE,
        "item_count": len(items),
        "corpus_fingerprint": corpus_fingerprint(items),
        # --- traceability only ---
        "source_json": JSON_PATH.name,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
    }


run_metadata = build_run_metadata(corpus)
print("fingerprint:", run_metadata["corpus_fingerprint"][:32], "...")
print("items      :", run_metadata["item_count"])


# %% [markdown]
# ### Cache check
# 
# Returns the reasons a rebuild is needed, rather than a bare boolean, so a rebuild
# always explains itself.
# 

# %%
def cache_status(current: dict) -> tuple[bool, list[str]]:
    """Is the cache on disk usable for this run? If not, why not."""
    reasons = []

    for path, label in ((VECTORS_PATH, "vectors"), (IDS_PATH, "ids"),
                        (METADATA_PATH, "metadata")):
        if not path.exists():
            reasons.append(f"no {label} file at {path.name}")
    if reasons:
        return False, reasons

    try:
        saved = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"metadata file is unreadable ({exc})"]

    for key in CACHE_KEYS:
        old, new = saved.get(key), current[key]
        if old != new:
            if key == "corpus_fingerprint":
                reasons.append("corpus changed (ids or text differ from the saved run)")
            else:
                reasons.append(f"{key} changed: {old!r} -> {new!r}")

    # The files must also agree with each other.
    if not reasons:
        saved_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        rows = np.load(VECTORS_PATH, mmap_mode="r").shape[0]
        if len(saved_ids) != rows:
            reasons.append(f"ids ({len(saved_ids)}) and vectors ({rows}) disagree")
        elif saved_ids != [i["id"] for i in corpus]:
            reasons.append("saved ids do not match the corpus order")

    return not reasons, reasons


is_cached, reasons = cache_status(run_metadata)
print("cache usable:", is_cached)
for reason in reasons:
    print("  -", reason)


# %% [markdown]
# ### Embed, or load from cache
# 
# This is the step the caching exists for. On a cold run the model is downloaded
# (~130 MB) and 901 items are encoded; on every later run the vectors are read
# straight from disk.
# 

# %%
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the model once per session and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _model


def compute_embeddings(items: list[dict], metadata: dict):
    """Encode every item and write vectors, ids and metadata to disk."""
    started = time.time()
    model = get_model()

    vectors = model.encode(
        [item[EMBED_FIELD] for item in items],
        batch_size=BATCH_SIZE,
        normalize_embeddings=NORMALIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(DTYPE)

    ids = [item["id"] for item in items]
    metadata = {
        **metadata,
        "dimension": int(vectors.shape[1]),
        "max_seq_length": int(model.max_seq_length),
        "encode_seconds": round(time.time() - started, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, vectors)
    IDS_PATH.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return vectors, ids, metadata


if is_cached:
    embeddings = np.load(VECTORS_PATH)
    embedding_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    print(f"Loaded cached embeddings, built {metadata.get('created_utc', 'unknown')}")
else:
    print("Rebuilding embeddings ...")
    embeddings, embedding_ids, metadata = compute_embeddings(corpus, run_metadata)
    print(f"Encoded in {metadata['encode_seconds']}s")

print(f"\nvectors {embeddings.shape} {embeddings.dtype}"
      f"  ({embeddings.nbytes / 1024 / 1024:.1f} MB)")


# %% [markdown]
# ###  Verification
# Cheap assertions, run every time. A silently misaligned index is the worst failure
# mode in a RAG system: it returns confident answers citing the wrong provision.
# 

# %%
assert embeddings.shape[0] == len(corpus), "one vector per item"
assert embeddings.shape[0] == len(embedding_ids), "one id per vector"
assert embedding_ids == [i["id"] for i in corpus], "ids align with corpus order"
assert embeddings.dtype == np.dtype(DTYPE), f"vectors should be {DTYPE}"
assert np.isfinite(embeddings).all(), "no NaN or inf values"

norms = np.linalg.norm(embeddings, axis=1)
if NORMALIZE:
    assert np.allclose(norms, 1.0, atol=1e-5), "vectors should be unit length"

# Index from item id to row, for looking up a provision's vector directly.
row_of = {item_id: row for row, item_id in enumerate(embedding_ids)}

print("all checks passed")
print(f"  items      {len(corpus)}")
print(f"  dimension  {embeddings.shape[1]}")
print(f"  L2 norms   min {norms.min():.6f}  max {norms.max():.6f}")
print(f"  model      {metadata['model_name']}")
print(f"  built      {metadata.get('created_utc', 'unknown')}")


# %% [markdown]
# ### Sequence length
# 
# `bge-small-en-v1.5` truncates at **512 tokens**. Anything longer is silently cut off,
# so it is worth knowing exactly which provisions are affected rather than discovering
# it through a bad answer later.
# 

# %%
# The model carries its own tokenizer, so this needs no extra download.
tokenizer = get_model().tokenizer
token_counts = np.array([len(tokenizer.encode(i[EMBED_FIELD])) for i in corpus])
limit = metadata.get("max_seq_length", 512)
over = token_counts > limit

print(f"token limit: {limit}")
for pct in (50, 90, 99, 100):
    print(f"  p{pct:<4} {int(np.percentile(token_counts, pct)):5} tokens")
print(f"\ntruncated items: {over.sum()} of {len(corpus)} ({100 * over.mean():.1f}%)")

for idx in np.argsort(-token_counts)[:5]:
    if over[idx]:
        print(f"  {token_counts[idx]:5} tokens  {corpus[idx]['id']:16} "
              f"{corpus[idx]['embed_text'][:46]}")


# %% [markdown]
# ## 3. Processing User Query

# %% [markdown]
# ### Searching for the most relevant information to the provided question
# 
# A hit on `art_6.para_2` alone is a fragment: it reads "In addition to the high-risk AI
# systems referred to in paragraph 1..." and the reader never sees paragraph 1. So each
# hit is expanded to the whole provision it belongs to — every paragraph of that article,
# or every point of that annex — in document order.
# 
# Three rules keep that from exploding:
# 
# - **Definitions never expand.** A hit on one definition would otherwise drag in all 68
#   of Article 3, roughly 4,500 tokens of mostly irrelevant text.
# - **Recitals expand only across their own chunks**, so a recital split in two is
#   reassembled rather than delivered half-read.
# - **Oversized groups fall back to the hits alone.** Rather than hardcoding which
#   provisions "don't make sense" to expand, any group above `MAX_GROUP_TOKENS`
#   contributes only the items that actually matched.
# 
# A total `CONTEXT_BUDGET` caps the whole context, and groups are added best-scoring
# first, so if anything is dropped it is the least relevant material.

# %%
# How much context to hand the model.
MAX_GROUP_TOKENS = 3000     # a single article/annex larger than this will not expand
CONTEXT_BUDGET = 12000      # ceiling for the whole assembled context


# Context Search Function
def search(question: str, k: int = 10) -> list[tuple[float, dict]]:
    query = get_model().encode([QUERY_PREFIX + question],
                         normalize_embeddings=NORMALIZE,
                         convert_to_numpy=True).astype(DTYPE)
    scores = embeddings @ query[0]
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), corpus[i]) for i in top]

# Determine the group to which an item belongs.
def group_of(item: dict):
    if item["type"] == "definition":
        return None
    if item["type"] == "article":
        return ("article", item["article"])
    if item["type"] == "annex":
        return ("annex", item["annex"])
    if item["type"] == "recital":
        return ("recital", item["recital"])
    return None


# Every group's rows, in document order. Built once.
GROUP_ROWS = defaultdict(list)
for _row, _item in enumerate(corpus):
    _key = group_of(_item)
    if _key is not None:
        GROUP_ROWS[_key].append(_row)

# Retrieving k items and expanding each to its whole provision, while respecting token limits.
def build_context(question: str, k: int = 10,
                  max_group_tokens: int = MAX_GROUP_TOKENS,
                  budget: int = CONTEXT_BUDGET) -> list[str]:
    # Performing the search and retrieving the top k hits.
    hits = search(question, k)

    # Collapsing the hits into groups, keeping best-score order.
    groups = OrderedDict()
    for score, item in hits:
        key = group_of(item) or ("item", item["id"])     # definitions stand alone
        if key not in groups:
            groups[key] = {"score": score, "rows": []}
        groups[key]["rows"].append(row_of[item["id"]])

    # Select groups to include, respecting the token budget.
    selected, used = [], 0
    for key, info in groups.items():
        rows = GROUP_ROWS.get(key, info["rows"])
        cost = int(token_counts[rows].sum())

        # A group too large to justify: fall back to the matching items only.
        if cost > max_group_tokens:
            rows = sorted(set(info["rows"]))
            cost = int(token_counts[rows].sum())

        if used + cost > budget:
            continue                                     # keep the better groups whole

        selected.extend(rows)
        used += cost

    # Document order overall, so the model reads the act as it is written.
    selected = sorted(set(selected))
    return [corpus[row]["embed_text"] for row in selected]

def build_prompt(question: str, context: list[str]) -> str:
    # Joining the context list into one string, so the prompt does not receive
    # Python's list repr (brackets, quotes and escapes) around every provision.
    provisions = "\n\n".join(context)

    query = f'''
You are an assistant answering questions about Regulation (EU) 2024/1689
(the EU AI Act).

Rules:
- Answer ONLY from the provisions given below. Do not use outside knowledge.
- Articles and Annexes are binding provisions. Recitals are interpretive
  context from the preamble and are NOT binding — use them to explain intent,
  never as the source of a rule or obligation.
- Cite the provision immediately after each claim, formatted as
  "Article 3(1)", "Annex III, point 4" or "Recital 12".
- If the provisions below do not answer the question, say so plainly rather
  than inferring an answer.
- Be concise. End with a "Sources:" line listing the provisions you used.
- This is informational only and is not legal advice.

QUESTION: {question}

PROVISIONS:
{provisions}

Now answer the question above, following the rules.
    '''
    return query


# %% [markdown]
# ## 4. Prompting the model

# %%
SEPARATOR = "----------------------------------------------------------------------------------------------------------------------------------------------"

# Tracking the last interaction so the model keeps context across questions.
previous_id = None

while True:
    print(SEPARATOR)

    # Asking for a question. Passing the prompt into input() keeps the question on the
    # same line as the prompt, rather than echoing it back separately.
    question = input("\nEnter a question (or 'exit' to quit): ").strip()

    # Checking if the user wants to exit the loop
    if question.lower() == "exit":
        break

    print(f"User question: {question}")

    print("Searching for relevant provisions ...")

    # Building the context based on the user's question
    context = build_context(question)

    # Building the prompt to send to the model
    prompt = build_prompt(question, context)

    # previous_interaction_id is only valid once there is an earlier turn to chain to.
    request = {"model": LLM_MODEL, "input": prompt, "stream": True}
    if previous_id is not None:
        request["previous_interaction_id"] = previous_id

    stream = client.interactions.create(**request)

    print()

    # Streaming the answer. The model's reasoning arrives as step.delta events too, but
    # carrying a ThoughtSignatureDelta instead of a TextDelta, so only text is printed.
    for event in stream:
        if event.event_type == "step.delta" and getattr(event.delta, "type", None) == "text":
            print(event.delta.text, end="", flush=True)

        # The interaction id travels on the events, not on the Stream object itself.
        elif event.event_type == "interaction.completed":
            previous_id = event.interaction.id

    print()


