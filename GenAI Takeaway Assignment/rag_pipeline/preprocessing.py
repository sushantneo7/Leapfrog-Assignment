import pandas as pd
from typing import List, Dict, Tuple
from .utils import clean_html, concat_fields, chunk_text

def load_and_prepare(path: str, chunk_size: int, overlap: int) -> Tuple[List[str], List[Dict]]:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type. Use CSV/XLSX.")

    # Normalize column names we expect
    rename_map = {
        "Title": "Job Title",
        "Company": "Company Name",
        "Category": "Job Category",
        "Seniority": "Level",
        "Description": "Job Description",
    }
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    chunks, metadatas = [], []
    for i, row in enumerate(df.fillna("").to_dict(orient="records")):
        job_id = row.get("Job ID") or row.get("ID") or row.get("Id") or f"AUTO-{i}"
        title = row.get("Job Title") or ""
        company = row.get("Company Name") or ""
        location = row.get("Location") or ""
        category = row.get("Job Category") or row.get("Category") or ""
        level = row.get("Level") or ""
        tags = row.get("Tags") or ""

        full = concat_fields(row)
        parts = chunk_text(full, chunk_size, overlap)
        for j, p in enumerate(parts):
            chunks.append(p)
            metadatas.append({
                "id": str(job_id),
                "chunk_id": f"{job_id}-{j}",
                "title": title,
                "company": company,
                "location": location,
                "category": category,
                "level": level,
                "tags": tags
            })
    return chunks, metadatas
