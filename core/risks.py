# core/risks.py
import re

RISK_KEYWORDS = {
    "Late Payments / Financial Penalties": r"late fee|penalt(y|ies)|interest|default in payment|overdue|delayed payment",
    "Termination & Breach": r"termination|breach of contract|void|cancellation|material breach|contract violation",
    "Confidentiality & Disclosure": r"disclosure|confidential|nda|non-disclosure|trade secret",
    "Liability & Indemnification": r"liable|indemnif(y|ication)|responsibility|hold harmless|compensation liability",
    "Intellectual Property Risks": r"intellectual property|copyright|patent|trademark|IP rights|proprietary|design rights",
    "Dispute Resolution (Indian Context)": r"arbitration|jurisdiction|venue|dispute resolution|litigation|mediation|conciliation",
    "Damages & Remedies": r"damages|compensation|losses|remedies|consequential damages|punitive damages|specific performance",
    "Defamation / Reputation": r"defamation|reputation|libel|slander|character assassination",
    "Automatic Renewal": r"auto-?renewal|automatic renewal|renewal term|extension of contract",
    "Legal Fees & Costs": r"advocate fees|legal fees|court costs|litigation expenses",
    "Employment / Labour Risks": r"non-compete|non compete|non-solicit|non solicit|employee|employment|labour|industrial dispute",
    "Governing Law / Jurisdiction (India)": r"governing law|laws of India|Indian Penal Code|IPC|jurisdiction of.{0,50}India|court of.{0,50}India|Supreme Court|High Court|Arbitration and Conciliation Act",
}


def analyze_risks(text: str) -> dict:
    
    risks = {}

    for category, pattern in RISK_KEYWORDS.items():
        matches = []
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            snippet = text[start:end].replace("\n", " ").strip()
            matches.append(snippet)
        if matches:
            risks[category] = list(set(matches))

    return risks if risks else {"No obvious risks detected": []}


def extract_risk_categories(text: str) -> str:

    matched = []

    for category, pattern in RISK_KEYWORDS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched.append(category)

    return ", ".join(matched) if matched else "None"