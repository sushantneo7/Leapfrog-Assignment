from bs4 import BeautifulSoup
import re
from typing import Dict, List

def clean_html(html_text: str) -> str:
    if not isinstance(html_text, str):
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def concat_fields(row: Dict) -> str:
    parts = []
    for key in ["Job Title","Company Name","Location","Job Category","Level","Tags","Job Description"]:
        val = row.get(key) or row.get(key.replace(" ", "_")) or row.get(key.lower())
        if val:
            if key == "Job Description":
                val = clean_html(val)
            parts.append(f"{key}: {val}")
    return " | ".join(parts)

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def parse_query_to_filters(q: str) -> Dict[str, str]:
    dq = (q or "").lower()
    filters = {}

    # LEVEL 
    level_keywords = {
        "Internship": ["internship", "intern"],
        "Entry": ["entry", "entry-level"],
        "Junior": ["junior"],
        "Mid": ["mid", "mid-level"],
        "Senior": ["senior"],
        "Lead": ["lead"],
        "Manager": ["manager"],
        "Principal": ["principal"]
    }
    for label, keywords in level_keywords.items():
        if any(kw in dq for kw in keywords):
            filters["level"] = label
            break

    # LOCATION 
    location_keywords = {
        "Remote": ["remote", "work from home"],
        "Onsite": ["onsite", "on-site", "in-office"],
        "Hybrid": ["hybrid", "flexible"]
    }
    for label, keywords in location_keywords.items():
        if any(kw in dq for kw in keywords):
            filters["location"] = label
            break

    # CATEGORY 
    category_keywords = {
        "Data": ["data", "data science", "analytics"],
        "AI": ["ai", "artificial intelligence"],
        "ML": ["ml", "machine learning"],
        "NLP": ["nlp", "natural language processing"],
        "Platform": ["platform"],
        "Backend": ["backend", "back end"],
        "Frontend": ["frontend", "front end"],
        "Mobile": ["mobile", "ios", "android"],
        "Security": ["security", "infosec", "cybersecurity"]
    }
    for label, keywords in category_keywords.items():
        if any(re.search(rf"\b{kw}\b", dq) for kw in keywords):
            filters["category"] = label
            break

    # COMPANY
    # Capture after "at" or "@"
    m = re.search(r"\b(?:at|@)\s+([A-Za-z][\w\-\.\s]+)", dq)
    if m:
        filters["company"] = m.group(1).strip()

    return filters