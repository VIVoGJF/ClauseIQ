# core/summarization.py
import re
import json
import ast
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# -------------------------
# Load API key
# -------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found. Please set it in .env or environment variables.")

# -------------------------
# LLM Wrapper
# -------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_retries=2,
    api_key=api_key
)

# -------------------------
# Helpers
# -------------------------
def _remove_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text

def _extract_first_json_block(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else text

def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        repaired = re.sub(r"(?<!\\)\'", '"', text)
        return json.loads(repaired)
    except Exception:
        return None

# -------------------------
# Main Summarizer
# -------------------------
def summarize_document(text: str, entities: dict, risks: dict, max_attempts: int = 2) -> str:
    """
    Passes extracted text/entities/risks to Gemini and returns a clean JSON string.
    """

    prompt = f"""
    You are a legal assistant.

    Here are extracted details from a legal document:

    --- Summary ---
    {text}

    --- Entities ---
    {json.dumps(entities, ensure_ascii=False)}

    --- Risks ---
    {json.dumps(risks, ensure_ascii=False)}

    Task:
    1. Fix grammar and improve readability of the summary and other entities.
    2. Add a "suggestion" field (1–3 sentences of advice).
    3. Return a single JSON with keys:
       - summary(5-7 lines)
       - parties(only valid parties(reffering to institue, individual,or group of individuals), no duplicates(like abreviations) or irrelevant entities)
       - date/time(
           mention context of a specific date/time with format "<b>context:</b> Value"
           example: 
                <b>Age of the cat:</b> 7 months
                <b>Date of the incident:</b> 2022-01-01
                <b>Document valid until:</b> 2023-01-01
            if any number is not a valid date with context unrelated to date/time, do not include it
            with each date/time on a new line, separated by <br/> inside the JSON string
            )
       - money/penalties
       - obligations
       - risks (string with multiple lines. Each risk must be formatted as:  
            "<b>• Risk Name</b>: Risk description(1line) asper Indian law" 
            with each risk on a new line, se parated by <br/> inside the JSON string)
       - suggestion
    ONLY return valid JSON. Do not include triple backticks or any extra text.
    ALWAYS check for any spelling misinterpretation of entities and correct them(
        example: sometimes extracted text from pdf can refer "KIIT" as "KUT" or "valid" as "valicl".
        such things should be reanalysed and fixed throughout the document).
    the entities (parties, date/time, money/penalties, obligations) as well as risks should return plain text, not inside [].
    if data is not available for any of the fields,return few words that might be relevant insted of none.
    DO NOT return any confidential information like party phone numbers, addresses(email can be returned as it necessary for further communication).
    """

    response_text = None
    last_cleaned = ""

    for attempt in range(max_attempts):
        try:
            resp = llm.invoke(prompt)
            response_text = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
        except Exception as e:
            print(f"[Groq error attempt {attempt+1}]: {e}")
            continue

        if not response_text:
            continue

        cleaned = _remove_code_fences(response_text)
        cleaned = _extract_first_json_block(cleaned).strip()
        parsed = _try_parse_json(cleaned)

        if parsed is not None:
            return json.dumps(parsed, ensure_ascii=False)
        last_cleaned = cleaned

    # fallback if still not valid JSON
    return json.dumps({
        "raw_output": response_text or last_cleaned,
        "error": "Not valid JSON"
    }, ensure_ascii=False)