import os
import cohere
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

def build_summary_prompt(query: str, jobs: list) -> str:
    """
    Build a structured prompt to guide the LLM.
    Keeps responses concise, accurate, and grounded.
    """

    if not jobs:
        return f"""
You are a helpful assistant for job search.
User query: "{query}"
No jobs were retrieved.
Instruction: Clearly state that no relevant jobs were found.
"""

    # Format retrieved jobs as clean bullet points
    job_str = "\n".join(
        f"- {job.get('title', 'Unknown Title')} at {job.get('company', 'Unknown Company')} ({job.get('location', 'N/A')})"
        for job in jobs
    )

    return f"""
You are a helpful assistant that summarizes job search results.

User query: "{query}"

Here are the most relevant jobs:
{job_str}

Instructions:
- Write a concise, factual summary (2–4 sentences max).
- Mention how many jobs were found.
- If the list is empty, ONLY then say: "Sorry, but there do not appear to be any relevant jobs." Otherwise, always summarize the jobs listed above.
- If title and company are available: list them normally.
- If some fields are missing: do NOT write "Unknown Title" or "Unknown Company".
- Instead, phrase naturally, e.g. "These are the listings that seem relevant to your query".
- Do NOT invent jobs or details beyond those listed above.
- Keep the tone professional and neutral.
- If no jobs are listed, clearly state "No relevant jobs found."
"""

def summarize(query: str, jobs: list) -> str:
    """
    Generate a natural language summary of retrieved jobs.
    Uses Cohere LLM if API key is available, else falls back to template.
    """

    # If no jobs at all → handle gracefully
    if not jobs:
        return f"No relevant jobs found for '{query}'."

    # Build prompt using only top 5 jobs for efficiency
    prompt = build_summary_prompt(query, jobs[:5])

    if client:
        try:
            response = client.generate(
                model="command",
                prompt=prompt,
                max_tokens=200,
                temperature=0.3,   # low temp = concise, accurate
            )
            return response.generations[0].text.strip()
        except Exception as e:
            # If Cohere fails, fallback to template
            print("Cohere summarization failed:", e)

    # -------- Template fallback --------
    job_list = ", ".join(
        f"{j.get('title', 'Unknown')} at {j.get('company', 'Unknown')}"
        for j in jobs[:3]
    )
    return f"Found {len(jobs)} relevant jobs for '{query}'. Top results: {job_list}."

