# core/entities.py
import re
import spacy

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "SpaCy model not found. Install with: python -m spacy download en_core_web_sm"
    )

# -------------------------
# Date patterns
# -------------------------
_DATE_PATTERNS = [
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
    r"\b\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b",
    r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*,?\s+\d{2,4}\b",
    r"\b\d+\s+(days?|weeks?|months?|years?)\b",
    r"\b[\(\[\{<\-*]?\d+[\)\]\}>\-*]?\s+(days?|weeks?|months?|years?)\b",
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(days?|weeks?|months?|years?)\b",
]


def extract_dates(text: str) -> list[str]:
    """
    HTML-formatted dates with context snippets.
    Used by summarization.py → pdf_writer.py pipeline.
    """
    contextual_dates = []
    for pattern in _DATE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            date_val = match.group()
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context_snippet = text[start:end].replace("\n", " ").strip()
            contextual_dates.append(f"<b>{context_snippet}:</b> {date_val}")

    seen = set()
    contextual_dates = [x for x in contextual_dates if not (x in seen or seen.add(x))]
    return contextual_dates if contextual_dates else ["Not specified"]


def extract_dates_clean(text: str) -> list[str]:
    """
    Plain date strings with no HTML — used for Pinecone chunk metadata.
    """
    dates = []
    for pattern in _DATE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            dates.append(match.group())

    seen = set()
    dates = [x for x in dates if not (x in seen or seen.add(x))]
    return dates if dates else ["Not specified"]


# -------------------------
# Entity extraction
# -------------------------
def extract_entities(text: str) -> dict:
    """
    Full entity extraction with HTML-formatted dates.
    Used by summarization.py pipeline.
    """
    doc = nlp(text)

    parties = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON"]]
    money = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

    # Cap match length to avoid greedy runaway on long clauses
    obligations = re.findall(
        r"\b(shall|must|agree to|responsible for)\b.{0,200}?\.",
        text,
        flags=re.IGNORECASE,
    )

    entities = {
        "Parties": list(set(parties)) or ["Not specified"],
        "Dates": extract_dates_clean(text),
        "Money/Penalties": list(set(money)) or ["Not specified"],
        "Obligations": obligations or ["Not specified"],
    }
    
    return entities


def extract_entities_clean(text: str) -> dict:
    """
    Plain entity extraction with no HTML — used for Pinecone chunk metadata.
    """
    doc = nlp(text)

    parties = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON"]]
    money = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

    obligations = re.findall(
        r"\b(shall|must|agree to|responsible for)\b.{0,200}?\.",
        text,
        flags=re.IGNORECASE,
    )

    entities = {
        "Parties": list(set(parties)) or ["Not specified"],
        "Dates": extract_dates_clean(text),
        "Money/Penalties": list(set(money)) or ["Not specified"],
        "Obligations": obligations or ["Not specified"],
    }
    
    return entities