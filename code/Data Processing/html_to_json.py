# %% [markdown]
# # EU AI Act — HTML to JSON
# 
# Parses the English EUR-Lex HTML of Regulation (EU) 2024/1689 (the AI Act)
# into a flat JSON list, one item per paragraph, to be used in a RAG-based question-answering system.
# 
# Source: `data/HTML/EU_AI_Act_EN.html` → Output: `data/JSON/eu_ai_act.json`
# 
# ```json
# {
#   "id": "art_6.para_2",
#   "type": "article",
#   "chapter": "III",
#   "section": "1",
#   "article": 6,
#   "paragraph": 2,
#   "text": "In addition to the high-risk AI systems referred to in paragraph 1, ...",
#   "article_title": "Classification rules for high-risk AI systems"
# }
# ```
# 

# %% [markdown]
# ## 1. Setup
# 

# %%
import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "HTML").is_dir():
            return candidate
    raise FileNotFoundError(f"No data/HTML directory found above {start}")


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
HTML_PATH = PROJECT_ROOT / "data" / "HTML" / "EU_AI_Act_EN.html"
JSON_PATH = PROJECT_ROOT / "data" / "JSON" / "eu_ai_act.json"

print("source:", HTML_PATH)
print("output:", JSON_PATH)


# %% [markdown]
# ## 2. How the source is marked up
# 
# EUR-Lex publishes the act as ELI-annotated XHTML. Four things matter:
# 
# | Markup | Meaning | Example |
# |---|---|---|
# | `div.eli-subdivision` | a recital or an article | `rct_99`, `art_6` |
# | `div` with a numeric id | one numbered paragraph | `006.002` = Article 6(2) |
# | `div.eli-title` | article / chapter / section heading | `art_6.tit_1`, `cpt_III.sct_1.tit_1` |
# | `div.eli-container` | main body, then one per Annex | `anx_III` |

# %%
# Initializing BeautifulSoup to parse the HTML content of the document (turns it into a navigable tree structure)
soup = BeautifulSoup(HTML_PATH.read_text(encoding="utf-8"), "lxml")

# Inline superscript footnote refs would inject stray digits into the text;
# footnote bodies at the end of the document belong to no paragraph.
for tag in soup.select("span.oj-super, p.oj-note"):
    tag.decompose()

# Selecting the main content within the document
articles = soup.select("div.eli-subdivision[id^='art_']")
recitals = soup.select("div.eli-subdivision[id^='rct_']")
containers = soup.select(".eli-container")
annexes = containers[1:]

# Printing the number of elements found for verification
print(f"Found {len(articles)} articles, {len(recitals)} recitals, and {len(annexes)} annexes.")

# %% [markdown]
# ## 3. Turning markup into text
# 

# %%
# Compiling regex patterns for identifying different types of IDs in the document

# Paragraph ID pattern: string begins with three digits, followed by a dot, 
# followed by three more digits (e.g., "001.001").
PARA_ID = re.compile(r"^(\d{3})\.(\d{3})$")

# Article ID pattern: string begins with "art_" followed by one or more digits (e.g., "art_1").
ARTICLE_ID = re.compile(r"^art_(\d+)$")

# Recital ID pattern: string begins with "rct_" followed by one or more digits (e.g., "rct_1").
RECITAL_ID = re.compile(r"^rct_(\d+)$")

# Chapter title ID pattern: string begins with "cpt_" followed by Roman numerals, 
# optionally followed by ".sct_" and one or more digits, and ending with ".tit_1" 
# (e.g., "cpt_I.sct_1.tit_1").
CHAPTER_TITLE_ID = re.compile(r"^cpt_([IVXLC]+)(?:\.sct_(\d+))?\.tit_1$")

# %%
# Returns the "id" attribute of a tag or an empty string if none exists
def eid(tag) -> str:
    return (tag.get("id") or "").strip()

# Cleans up text
def clean(text: str) -> str:
    # Collapsing non-breaking spaces and other whitespace characters into regular spaces
    text = text.replace("\xa0", " ").replace("\u2007", " ").replace("\u2009", " ")

    # Normalizing Unicode characters to their canonical form (NFC) to ensure consistent representation
    text = unicodedata.normalize("NFC", text)

    # Collapsing every run of whitespace -- spaces, tabs and newlines alike --
    # into a single space, so no text field ever carries a line break
    return re.sub(r"\s+", " ", text).strip()

# Flattens the text of a single HTML element, cleaning it up for further processing
def flat_text(node) -> str:
    return clean(node.get_text(" ", strip=True))

# Returns the direct rows of a table, handling the presence of an optional <tbody> element
def table_rows(table):
    """Direct rows of a table, tolerating an optional <tbody>."""
    body = table.find("tbody", recursive=False)
    return (body or table).find_all("tr", recursive=False)

# Flattens EUR-Lex markup into indented lines
def extract_blocks(node, depth: int = 0) -> list[str]:
    # Defining a list to hold the extracted lines
    lines: list[str] = []

    # Calculating the padding based on the current depth in the document hierarchy
    pad = "  " * depth

    # Iterating through the direct children of the current node
    for child in node.find_all(recursive=False):
        # Checking for table elements to extract rows and cells
        if child.name == "table":
            # Iterating through each row in the table
            for row in table_rows(child):
                # Extracting the cells of the current row
                cells = row.find_all("td", recursive=False)

                # If there are no cells in the row, skip to the next iteration
                if not cells:
                    continue

                # Initializing a marker to hold any text from the cells except the last one
                marker = ""

                # Iterating through all cells except the last one to find a marker text
                for cell in cells[:-1]:
                    # If the cell has text, set it as the marker and break the loop
                    if flat_text(cell):
                        marker = flat_text(cell)

                # Recursively extracting blocks from the last cell of the row, increasing the depth for indentation
                sub = extract_blocks(cells[-1], depth + 1)

                # If the recursive extraction returns no lines, skip to the next iteration
                if not sub:
                    continue

                # If a marker was found, prepend it to the first line of the extracted sub-blocks; otherwise, just add the first line
                head = sub[0].strip()
                lines.append(f"{pad}{marker} {head}".strip() if marker else f"{pad}{head}")
                lines.extend(sub[1:])

        # Handling other block-level elements like div, tbody, tr, and td by recursively extracting their content
        elif child.name in ("div", "tbody", "tr", "td"):
            lines.extend(extract_blocks(child, depth))

        # Handling inline elements like paragraphs, spans, and list items by flattening their text
        elif child.name in ("p", "span", "li"):
            text = flat_text(child)
            if text:
                lines.append(f"{pad}{text}")

    # If no lines were extracted from the current node, check if it's a leaf node or an element whose only children are inline tags. 
    # In such cases, flatten the text of the node and add it to the lines if it's not empty.
    if not lines:
        text = flat_text(node)
        if text:
            lines.append(f"{pad}{text}")

    return lines

# Flattens EUR-Lex markup into a single line of text
def block_text(node) -> str:
    # Joining with a space rather than a newline keeps each provision on one line;
    # the (a)/(b) markers still mark the structure inline
    return re.sub(r"\s+", " ", " ".join(extract_blocks(node))).strip()


# %% [markdown]
# ## 4. Chapters, sections and titles
# 
# Walking the enacting-terms container in document order and remembering the most
# recent heading gives every article its chapter and section.
# 

# %%
def build_chapter_map(soup) -> dict:
    # Mapping of article IDs to their corresponding chapter and section identifiers
    mapping = {}

    # Initializing variables to keep track of the current chapter and section while iterating through the document
    chapter = section = None

    # Searching for the main content of the document, specifically looking for a div with id "enc_1" which represents the enacting terms
    enacting = soup.find("div", id="enc_1") or soup

    # Iterating through all div elements within the enacting terms
    for div in enacting.find_all("div"):

        # Extracting the classes of the current div element
        classes = div.get("class") or []

        # Using regex to match the chapter title ID pattern against the id of the current div element
        match = CHAPTER_TITLE_ID.match(eid(div))

        # If a match is found and the div has the class "eli-title", it indicates that this div represents a chapter title
        if match and "eli-title" in classes:
            chapter, section = match.group(1), match.group(2)   # section None -> reset
            continue

        # If the div has the class "eli-subdivision" and its id matches the article ID pattern, it indicates that this div represents an article
        if "eli-subdivision" in classes and ARTICLE_ID.match(eid(div)):
            mapping[eid(div)] = (chapter, section)

    return mapping


# Title text for every heading element, keyed by id (art_6.tit_1, cpt_III.tit_1, ...)
TITLES = {eid(d): block_text(d) for d in soup.find_all("div", class_="eli-title") if eid(d)}
CHAPTER_MAP = build_chapter_map(soup)

print("articles mapped to a chapter:", len(CHAPTER_MAP))
print("Article 6 ->", CHAPTER_MAP["art_6"], "|", TITLES["art_6.tit_1"])


# %% [markdown]
# ## 5. Parsing the articles
# 
# Most articles are split into numbered paragraphs, each its own item. Articles with single unnumbered bodies become one item with paragraph = null.
# 
# **Article 3 is the exception.** It holds all 68 definitions in one unnumbered body — 17,000 characters, nearly four times the next largest article. Left whole it would be useless for retrieval: a question about one defined term would drag in all 68. So it is split into one item per definition, typed `definition`, with the defined term lifted into its own field.

# %%
# Defining a helper function to create a dictionary representing an article item with its details
def item(paragraph, text, number, title, chapter, section):
    return {
        "id": f"art_{number}.para_{paragraph}" if paragraph else f"art_{number}",
        "type": "article",
        "chapter": chapter,
        "section": section,
        "article": number,
        "paragraph": paragraph,
        "text": text,
        "article_title": title,
    }

def parse_articles(soup) -> list[dict]:
    # Storing the parsed article items in a list
    items = []

    # Iterating through all div elements with the class "eli-subdivision" to extract article information
    for div in soup.find_all("div", class_="eli-subdivision"):

        # Using regex to match the article ID pattern against the id of the current div element
        match = ARTICLE_ID.match(eid(div))

        # If the current div does not match the article ID pattern, skip to the next iteration
        if not match:
            continue

        # Extract the article number from the matched ID
        number = int(match.group(1))

        # Retrieve the corresponding chapter and section for the current article from the CHAPTER_MAP
        chapter, section = CHAPTER_MAP.get(eid(div), (None, None))

        # Retrieve the title of the current article from the TITLES dictionary using its id
        title = TITLES.get(f"{eid(div)}.tit_1", "")

        # Finding all direct child div elements of the current article div that match the paragraph ID pattern
        paragraphs = [d for d in div.find_all("div", recursive=False) if PARA_ID.match(eid(d))]

        # If there are paragraphs found, iterate through each paragraph to extract its position and text content
        if paragraphs:
            for para in paragraphs:
                position = int(PARA_ID.match(eid(para)).group(2))
                # Drop the leading "2.   " -- redundant with the paragraph field.
                text = re.sub(rf"^{position}\s*[.)]\s*", "", block_text(para), count=1)
                if text:
                    items.append(item(position, text, number, title, chapter, section))

        # Article 3 (Definitions) is the one article worth splitting further
        elif number == 3:
            items.extend(parse_definitions(div, number, title, chapter, section))

        # If no paragraphs are found, treat the entire article div as a single block of text
        else:
            body = div.__copy__()
            for drop in body.find_all(["p", "div"], recursive=False):
                if {"oj-ti-art", "eli-title"} & set(drop.get("class") or []):
                    drop.decompose()
            text = block_text(body)
            if text:
                items.append(item(None, text, number, title, chapter, section))

    return items

# Regular expression to match the term in a definition
TERM = re.compile(r"^[\u2018\u201c'\"]([^\u2019\u201d'\"]+)[\u2019\u201d'\"]")

# Article 3 (Definitions) is the one article worth splitting further as its structure does not contain split paragraphs. 
def parse_definitions(div, number, title, chapter, section) -> list[dict]:
    # Storing the parsed definition items in a list
    items = []

    # The lead-in sentence that precedes the numbered list, kept so no text is lost.
    for lead in div.find_all("p", class_="oj-normal", recursive=False):
        text = flat_text(lead)
        if text:
            items.append({
                "id": f"art_{number}.intro",
                "type": "article",
                "chapter": chapter,
                "section": section,
                "article": number,
                "paragraph": None,
                "text": text,
                "article_title": title,
            })

    # Each definition is a direct-child table: a marker cell "(1)", then the body.
    for position, table in enumerate(div.find_all("table", recursive=False), start=1):
        cells = table.find("tr").find_all("td", recursive=False)
        text = block_text(cells[-1])
        if not text:
            continue

        # Prefer the number printed in the marker cell over the loop counter.
        marker = ""
        for cell in cells[:-1]:
            if flat_text(cell):
                marker = flat_text(cell)
        digits = re.sub(r"\D", "", marker)
        term = TERM.match(text)

        items.append({
            "id": f"art_{number}.def_{digits or position}",
            "type": "definition",
            "chapter": chapter,
            "section": section,
            "article": number,
            "paragraph": None,
            "definition": int(digits or position),
            "term": term.group(1) if term else None,
            "text": text,
            "article_title": title,
        })

    return items

articles = parse_articles(soup)
definitions = [i for i in articles if i["type"] == "definition"]
print(f"{len(articles)} items from articles, of which {len(definitions)} are definitions\n")
print(json.dumps(next(a for a in articles if a["id"] == "art_6.para_2"), indent=2))
print()
print(json.dumps(next(a for a in articles if a["id"] == "art_3.def_1"), indent=2))

# %% [markdown]
# ## 6. Recitals and Annexes
# 
# The 180 recitals carry the act's interpretive intent, and the Annexes carry the
# operative lists — Annex III alone defines the high-risk categories. Both are
# needed to answer questions about a project, so they are included using the same
# flat schema, distinguished by type.
# 

# %%
def parse_recitals(soup) -> list[dict]:
    # Storing the parsed recital items in a list
    items = []

    # Iterating through all div elements with the class "eli-subdivision" to extract recital information
    for div in soup.find_all("div", class_="eli-subdivision"):
        # Using regex to match the recital ID pattern against the id of the current div element
        match = RECITAL_ID.match(eid(div))
        if not match:
            continue

        # Extract the recital number from the matched ID
        number = int(match.group(1))

        # Drop the leading "(1)   " -- redundant with the recital field.
        text = re.sub(rf"^\(\s*{number}\s*\)\s*", "", block_text(div), count=1)

        # If the extracted text is not empty, create a dictionary representing the recital item and append it to the items list
        if text:
            items.append({
                "id": f"rct_{number}",
                "type": "recital",
                "recital": number,
                "text": text,
            })

    return items


# Annexes whose points are too small to stand on their own as retrieval units.
# Annex II lists criminal offences as bare terms -- "rape,", "sabotage," -- with a
# median length of 40 characters against 109+ for every other annex. Split out they
# retrieve as meaningless fragments, and Article 5(1) refers to the list as one
# closed set, so it is kept whole.
WHOLE_ANNEXES = {"II"}


def parse_annexes(soup) -> list[dict]:
    # Storing the parsed annex items in a list
    items = []

    # Iterating through all elements with the class "eli-container" to extract annex information
    containers = soup.find_all(class_="eli-container")

    # Skipping the first container as it represents the main body of the document, and iterating through the remaining containers which represent annexes
    for container in containers[1:]:
        # Extracting the annex identifier by removing the "anx_" prefix from the id of the current container
        roman = eid(container).removeprefix("anx_")

        # Finding all direct child paragraph elements of the current container that have the class "oj-doc-ti" to extract the annex title
        headings = container.find_all("p", class_="oj-doc-ti", recursive=False)

        # If there are at least two headings found, use the second one as the annex title; otherwise, set the title to an empty string
        title = flat_text(headings[1]) if len(headings) >= 2 else ""

        # Initializing lists to hold the introductory text and points of the annex, and a flag to indicate whether the parsing has started
        intro, points, started = [], [], False

        # Iterating through all direct child elements of the current container to extract the content of the annex
        for child in container.find_all(recursive=False):
            # Extracting the classes of the current child element
            classes = child.get("class") or []

            # If the current child is a paragraph with the class "oj-doc-ti", it represents the annex heading itself
            if child.name == "p" and "oj-doc-ti" in classes:
                continue                        

            # If the current child is a paragraph with the class "oj-ti-grseq-1", it indicates the start of a sub-heading (e.g., "Section A.")
            if child.name == "p" and "oj-ti-grseq-1" in classes:
                started = True                 
                continue

            # If the current child is a table, iterate through its rows to extract points of the annex
            if child.name == "table":
                for row in table_rows(child):
                    # Extracting the cells of the current row
                    cells = row.find_all("td", recursive=False)
                    if not cells:
                        continue

                    # If the current row has cells, extract the text from the last cell and add it to the points list if it's not empty
                    text = block_text(cells[-1])

                    # If the extracted text is not empty, append it to the points list and set the started flag to True
                    if text:
                        points.append(text)
                        started = True
                continue

            # Checking if the current child is a div with the class "oj-enumeration-spacing", 
            # which indicates that the annex points are numbered with plain divs rather than tables
            if child.name == "div" and "oj-enumeration-spacing" in classes:
                # Extracting the text from the current child div, removing any leading numbering (e.g., "1. ") using regex
                text = re.sub(r"^\d{1,3}\s*\.\s*\n?", "", block_text(child), count=1)

                # If the extracted text is not empty, append it to the points list
                if text:
                    points.append(text)
                    started = True
                continue

            # For any other child elements, extract their text content and add it to the appropriate list (intro or points)
            text = block_text(child)
            if not text:
                continue
            if started:
                # Where a group holds a single item EUR-Lex drops the table and
                # emits a bare paragraph; it is still a point.
                points.append(text)
            else:
                intro.append(text)

        # Annexes listed in WHOLE_ANNEXES are emitted as one item, intro and
        # points joined, rather than split point by point
        if roman in WHOLE_ANNEXES:
            text = " ".join(intro + points)
            if text:
                items.append({
                    "id": f"anx_{roman}", "type": "annex", "annex": roman,
                    "point": None, "text": text, "annex_title": title,
                })
            continue

        # If there is any introductory text collected, create a dictionary representing the annex introduction and append it to the items list
        if intro:
            items.append({
                "id": f"anx_{roman}.intro", "type": "annex", "annex": roman,
                "point": None, "text": " ".join(intro), "annex_title": title,
            })

        # If there are any points collected, iterate through each point to create a dictionary representing the annex point and append it to the items list
        for position, text in enumerate(points, start=1):
            items.append({
                "id": f"anx_{roman}.point_{position}", "type": "annex", "annex": roman,
                "point": position, "text": text, "annex_title": title,
            })

    return items

# Parsing recitals
recitals = parse_recitals(soup)

# Parsing annexes
annexes = parse_annexes(soup)

print(f"{len(recitals)} recitals, {len(annexes)} annex items")


# %% [markdown]
# ## 7. Text for embedding
# 
# Each item gets an `embed_text` field: a header naming the provision, then the text
# itself. Embedding that combined string rather than the bare text means a passage's
# vector carries **where it sits in the act** as well as what it says.
# 
# This matters because the text alone is often anonymous. Article 6(2) reads "In addition
# to the high-risk AI systems referred to in paragraph 1..." — nothing in it mentions
# Article 6, high-risk classification rules, or Chapter III. A question phrased as "which
# article classifies high-risk systems" has almost nothing to match against. With the
# header prepended, it does.
# 
# | Type | Header |
# |---|---|
# | article paragraph | `Article 6(2) — Classification rules for high-risk AI systems [Chapter III, Section 1]` |
# | definition | `Article 3(1) — Definitions: 'AI system' [Chapter I]` |
# | recital | `Recital 52 (AI Act preamble)` |
# | annex point | `Annex III, point 4 — High-risk AI systems referred to in Article 6(2)` |
# | whole annex | `Annex II — List of criminal offences referred to in Article 5(1)...` |

# %%
def citation(item) -> str:
    """A short header locating the provision within the act."""

    # Recitals sit in the preamble, outside the chapter and article structure
    if item["type"] == "recital":
        header = f"Recital {item['recital']} (AI Act preamble)"
        return header + (f" [part {item['chunk']}]" if item.get("chunk") else "")

    # Annexes are located by roman numeral and, where split, by point number
    if item["type"] == "annex":
        header = f"Annex {item['annex']}"
        if item["point"] is not None:
            header += f", point {item['point']}"
        if item["annex_title"]:
            header += f" \u2014 {item['annex_title']}"
        return header + (f" [part {item['chunk']}]" if item.get("chunk") else "")

    # Articles and definitions are both cited as "Article N(x)"
    header = f"Article {item['article']}"
    if item["type"] == "definition":
        header += f"({item['definition']})"
    elif item["paragraph"] is not None:
        header += f"({item['paragraph']})"

    # A paragraph split into its points cites the point too: "Article 5(1)(f)"
    if item.get("point") is not None:
        header += f"({item['point']})"

    if item["article_title"]:
        header += f" \u2014 {item['article_title']}"

    # The defined term is the thing a reader would actually search for
    if item["type"] == "definition" and item.get("term"):
        header += f": \u2018{item['term']}\u2019"

    # Chapter and section place the article within the structure of the act
    if item["chapter"]:
        where = f"Chapter {item['chapter']}"
        if item["section"]:
            where += f", Section {item['section']}"
        header += f" [{where}]"

    if item.get("chunk"):
        header += f" [part {item['chunk']}]"

    return header


def build_embed_text(item) -> str:
    """The header, then the provision's text, as one string to embed."""
    return f"{citation(item)} {item['text']}"


# Previewing the header produced for one item of each kind
for sample_id in ("art_6.para_2", "art_3.def_1", "rct_52", "anx_III.point_4", "anx_II"):
    sample = next(i for i in articles + recitals + annexes if i["id"] == sample_id)
    print(f"{sample_id:16} {citation(sample)}")

# %% [markdown]
# ## 8. Fitting the embedding model's token window
# 
# The corpus is embedded with `BAAI/bge-small-en-v1.5`, which reads at most **512
# tokens** and silently discards the rest — it does not drop the item, it returns a
# confident vector describing only the opening. That is the more dangerous failure:
# nothing errors, and the index just quietly misses things.
# 
# Twenty-one items exceeded the window, the worst being Article 5(1) at 950 tokens.
# Only points (a) to (e) survived; **(f) emotion recognition, (g) biometric
# categorisation and (h) live facial recognition were never embedded at all.**
# 
# Raising the limit is not an option — the model has learned position vectors for
# exactly 512 positions, and a larger BGE model has the same window. Splitting is also
# the better representation regardless: one 384-dimension vector cannot stand for eight
# distinct prohibitions.
# 
# Two strategies, applied in order:
# 
# 1. **Structural split.** Where a provision is a list of lettered points, split it into
#    one item per point, carrying the lead-in sentence so each stands alone.
#    `art_5.para_1` becomes `art_5.para_1.point_a` … `point_h`, citing as
#    *Article 5(1)(f)*. The split walks the source tables rather than pattern-matching
#    the flattened text, because markers like `(a)` also appear inside cross-references.
# 2. **Prose chunking.** Recitals are solid prose with no points, so they are divided on
#    sentence boundaries into the fewest chunks that fit.

# %%
from transformers import AutoTokenizer

# The window belongs to the embedding model, so the limit is defined by it.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 512
TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def n_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def top_level_points(element):
    """(lead_in, [(marker, body_element, body_text), ...]) for one element.

    Reads the source markup rather than the flattened text: markers like "(a)"
    also occur inside cross-references such as "referred to in points (a) and (b)",
    so pattern-matching the text would split in the wrong places.
    """
    lead, points = [], []
    for child in element.find_all(recursive=False):
        classes = child.get("class") or []

        if child.name == "p":
            if {"oj-ti-art", "oj-doc-ti"} & set(classes):
                continue                      # the heading, not body text
            text = flat_text(child)
            if text and not points:
                lead.append(text)             # prose before the first point
            continue

        if child.name == "div" and "eli-title" in classes:
            continue

        if child.name == "table":
            for row in table_rows(child):
                cells = row.find_all("td", recursive=False)
                if not cells:
                    continue
                marker = ""
                for cell in cells[:-1]:
                    if flat_text(cell):
                        marker = flat_text(cell)
                text = block_text(cells[-1])
                if text:
                    points.append((marker.strip("()., "), cells[-1], text))

    return " ".join(lead), points


def source_element(item):
    """The HTML element an item was parsed from, or None."""
    if item["type"] == "recital":
        return soup.find("div", id=f"rct_{item['recital']}")

    if item["type"] in ("article", "definition"):
        article = soup.find("div", id=f"art_{item['article']}")
        if article is None or item.get("paragraph") is None:
            return article
        # Scoped to the parent article: Articles 107-109 quote other regulations
        # and reuse their paragraph ids, so a document-wide lookup is ambiguous.
        return article.find("div", recursive=False,
                            id=f"{item['article']:03d}.{item['paragraph']:03d}")

    if item["type"] == "annex":
        container = soup.find(class_="eli-container", id=f"anx_{item['annex']}")
        if container is None or not str(item.get("point", "")).isdigit():
            return None
        _, points = top_level_points(container)
        index = int(item["point"]) - 1
        return points[index][1] if 0 <= index < len(points) else None

    return None


SENTENCE = re.compile(r"(?<=[.;:])\s+")


def chunk_prose(text, budget):
    """Divide prose on sentence boundaries into the fewest chunks that fit."""
    chunks, current = [], ""
    for sentence in SENTENCE.split(text):
        candidate = f"{current} {sentence}".strip()
        if current and n_tokens(candidate) > budget:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def derive(item, suffix, text, extra):
    """A child item, inheriting its parent's metadata."""
    child = {**item, "id": f"{item['id']}.{suffix}", "text": text, **extra}
    child["embed_text"] = build_embed_text(child)
    return child


def fit_item(item, element=None, depth=0):
    """Split one item until every piece fits inside the token window."""
    item = {**item, "embed_text": build_embed_text(item)}
    if n_tokens(item["embed_text"]) <= MAX_TOKENS:
        return [item]

    if element is None:
        element = source_element(item)

    # Strategy 1: split into the provision's own lettered points.
    if element is not None and depth < 3:
        lead, points = top_level_points(element)
        if item.get("paragraph") is not None:
            lead = re.sub(rf"^{item['paragraph']}\s*[.)]\s*", "", lead, count=1)

        if len(points) >= 2:
            pieces = []
            for marker, body, text in points:
                label = marker or str(len(pieces) + 1)
                # The lead-in travels with each point so it can be read alone.
                joined = f"{lead} ({label}) {text}".strip() if lead else f"({label}) {text}"
                child = derive(item, f"point_{label}", joined, {"point": label})
                pieces.extend(fit_item(child, body, depth + 1))
            return pieces

    # Strategy 2: prose with no points -- chunk on sentence boundaries.
    chunks = chunk_prose(item["text"], MAX_TOKENS - n_tokens(citation(item)) - 4)
    if len(chunks) == 1:
        return [item]                          # nothing more we can do
    return [derive(item, f"chunk_{n}", text, {"chunk": n})
            for n, text in enumerate(chunks, 1)]


def fit_corpus(items):
    """Apply the token limit across the whole corpus."""
    fitted = []
    for item in items:
        fitted.extend(fit_item(item))

    grew = len(fitted) - len(items)
    print(f"{len(items)} items -> {len(fitted)} after splitting (+{grew})")
    return fitted

# %% [markdown]
# ## 9. Build and save

# %%
# Combining all parsed items (articles, recitals, and annexes) into a single corpus
corpus = articles + recitals + annexes

# Stamping the embedding text onto every item, so the vector for a passage carries
# where it sits in the act as well as what it says
for entry in corpus:
    entry["embed_text"] = build_embed_text(entry)

# Splitting anything that would not survive the embedding model's 512-token window
corpus = fit_corpus(corpus)

# Saving the combined corpus as a JSON file
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
JSON_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

# JSON information
print(f"{len(corpus)} items -> {JSON_PATH} ({JSON_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
for kind in ("article", "definition", "recital", "annex"):
    print(f"  {kind:11} {sum(1 for i in corpus if i['type'] == kind):4}")

# %% [markdown]
# ## 10. Checks
# 
# The act has 113 Articles, 180 Recitals, 13 Annexes and 68 definitions in Article 3.
# Anything else means the parse dropped something. Every item must also fit the
# embedding model's 512-token window.

# %%
# Performing sanity checks to ensure the integrity and completeness of the parsed data
# Article 3 items are typed "definition", so both types count towards article coverage
article_numbers = {i["article"] for i in corpus if i["type"] in ("article", "definition")}
recital_numbers = {i["recital"] for i in corpus if i["type"] == "recital"}
annex_romans = {i["annex"] for i in corpus if i["type"] == "annex"}
definitions = [i for i in corpus if i["type"] == "definition"]

print("Articles 1-113 present :", sorted(article_numbers) == list(range(1, 114)))
print("Recitals 1-180 present :", sorted(recital_numbers) == list(range(1, 181)))
print("Annexes present        :", len(annex_romans), sorted(annex_romans))
print("Definitions 1-68       :", sorted(d["definition"] for d in definitions) == list(range(1, 69)))
print("definitions with a term:", sum(1 for d in definitions if d["term"]), "/", len(definitions))
print("duplicate ids          :", len(corpus) - len({i["id"] for i in corpus}))
print("empty text             :", sum(1 for i in corpus if not i["text"].strip()))
print("items without a title  :", sum(1 for i in corpus
                                      if i["type"] in ("article", "definition")
                                      and not i["article_title"]))
print("longest single item    :", max(len(i["text"]) for i in corpus), "chars")

# Nothing may exceed the embedding model's context window
token_lengths = [n_tokens(i["embed_text"]) for i in corpus]
print("longest in tokens      :", max(token_lengths), f"(limit {MAX_TOKENS})")
print("items over the limit   :", sum(1 for n in token_lengths if n > MAX_TOKENS))

# Every item must carry an embed_text that actually starts with its header
print("items with embed_text  :", sum(1 for i in corpus if i.get("embed_text")), "/", len(corpus))
print("headers are unique     :", len({citation(i) for i in corpus}) == len(corpus))
print("embed_text ends in text:", all(i["embed_text"].endswith(i["text"]) for i in corpus))

# No text field may contain a line break, tab, or doubled space
print("newlines / tabs in text:", sum(1 for i in corpus
                                      for f in ("text", "embed_text")
                                      if "\n" in i[f] or "\t" in i[f]))
print("doubled spaces in text :", sum(1 for i in corpus
                                      for f in ("text", "embed_text")
                                      if "  " in i[f]))
print("untrimmed text         :", sum(1 for i in corpus
                                      for f in ("text", "embed_text")
                                      if i[f] != i[f].strip()))

# %%
# Spot-check the provisions a feasibility question is most likely to hit.
for target in ("art_6.para_2", "art_5.para_1.point_f", "anx_III.point_4", "art_3.def_1"):
    entry = next(i for i in corpus if i["id"] == target)
    print(f"\n--- {entry['id']} ---")
    print(entry["embed_text"][:300])


