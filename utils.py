# utils.py
import re
import json
from typing import Dict, Any, List, Optional
import pdfplumber
import pytesseract
from pdf2image import convert_from_path


def extract_lor_text(pdf_path: str) -> Optional[str]:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        if not text.strip():
            pages = convert_from_path(pdf_path)
            for page in pages:
                text += pytesseract.image_to_string(page)
    except Exception as e:
        print(f"Error reading LOR: {e}")
        return None
    return text.strip() if text else None


def _infer_academic_level_from_text(text: str) -> Optional[str]:
    """Very lightweight heuristic to guess academic level from free text."""
    t = text.lower()

    # High-school patterns
    if any(kw in t for kw in [
        "12th", "class 12", "class xii", "higher secondary", "plus two", "hsc",
        "intermediate", "puc", "pre-university"
    ]):
        # Avoid misclassifying people clearly talking about 'completed bachelor's'
        if "b.tech" not in t and "btech" not in t and "bachelor" not in t:
            return "high_school"

    # Undergrad patterns
    if any(kw in t for kw in [
        "b.tech", "btech", "b.e", "b.e.", "bachelor of", "bachelors in",
        "bsc", "b.sc", "bca", "bcom", "b.com", "ba ", "b.a "
    ]):
        return "undergraduate"

    # Postgrad patterns
    if any(kw in t for kw in [
        "m.tech", "mtech", "m.e", "master of", "ms in", "m.s.", "msc", "m.sc",
        "mba", "pgdm"
    ]):
        return "postgraduate"

    # Working professional hints
    if any(kw in t for kw in [
        "working", "work experience", "software engineer", "developer at",
        "currently employed", "full-time job"
    ]):
        return "working_professional"

    return None


def _extract_competitive_exams(text: str) -> Optional[List[Dict[str, str]]]:
    """
    Extract mentions of common competitive exams with a small context snippet.
    We don't over-parse; we just keep free-text details for downstream reasoning.
    """
    exam_pattern = r"\b(JEE(?:\s*Main|\s*Advanced)?|NEET|SAT|ACT|BITSAT|VITEEE|COMEDK|MHT[-\s]?CET|KCET|CUET)\b[^.\n]*"
    exams: List[Dict[str, str]] = []
    for m in re.finditer(exam_pattern, text, re.IGNORECASE):
        full_snippet = m.group(0).strip()
        exam_name = m.group(1).strip().upper()
        exams.append({
            "exam_name": exam_name,
            "details": full_snippet
        })
    return exams or None


def _extract_board(text: str) -> Optional[str]:
    t = text.lower()
    if "cbse" in t:
        return "CBSE"
    if "icse" in t:
        return "ICSE"
    if "state board" in t or "stateboard" in t:
        return "State Board"
    if "hsc" in t or "higher secondary" in t:
        return "HSC"
    if "puc" in t or "pre-university" in t:
        return "PUC"
    return None


def _extract_class12_score(text: str) -> Optional[str]:
    """
    Try to find a 12th/HS score as a percentage or marks.
    We keep it as free-text, not normalized.
    """
    # Look for patterns like "12th ... 92%" or "class 12 ... 480/500"
    patterns = [
        r"(?:12th|class 12|class xii|higher secondary|hsc)[^.\n%]*?(\d{2,3}(?:\.\d+)?)\s*%",
        r"(?:12th|class 12|class xii|higher secondary|hsc)[^.\n]*?(\d{2,3,4}\/\d{2,3,4})"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback: any percentage mentioned with "12th" nearby in the same line
    line_pattern = r".*(12th|class 12|class xii|higher secondary|hsc).*\n?"
    for line_match in re.finditer(line_pattern, text, re.IGNORECASE):
        line = line_match.group(0)
        perc = re.search(r"(\d{2,3}(?:\.\d+)?)\s*%", line)
        if perc:
            return perc.group(1).strip() + "%"

    return None


def extract_info_from_text(text: str) -> Dict[str, Any]:
    """
    Heuristic, non-LLM extractor from free text.
    This is intentionally *best-effort* and does NOT try to be complete.
    Missing fields are simply omitted; downstream logic can still use raw text.
    """
    info: Dict[str, Any] = {}

    # Basic fields via regex or keywords
    name_match = re.search(
        r"\b(?:i'?m|i am|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        text,
        re.IGNORECASE
    )
    if name_match:
        info["student_name"] = name_match.group(1).strip()

    degree_match = re.search(
        r"(b\.tech|btech|engineering|bachelor|bachelors|bsc|b\.?e\.?|ba|bcom|bca|computer science|information technology|mechanical|civil)",
        text,
        re.IGNORECASE
    )
    if degree_match:
        info["current_degree"] = degree_match.group(1).strip()

    # Year (grad / passing / exam year) – still best-effort
    year_match = re.search(
        r"(?:graduat|finish|complete|passed out|year)\s*(?:in|year)?\s*(\d{4}|\d{2})",
        text,
        re.IGNORECASE
    )
    if not year_match:
        # Fallback to any 4-digit year
        year_match = re.search(r"\b(\d{4})\b", text)
    if year_match:
        year = year_match.group(1).strip()
        if len(year) == 2:
            year = "20" + year  # Assume 20xx
        info["graduation_year"] = year

    cgpa_match = re.search(r"cgpa\s*(?:is|:|of)?\s*([\d.]+(?:\s*/\s*\d+(?:\.\d+)?)?)", text, re.IGNORECASE)
    if cgpa_match:
        info["cgpa"] = cgpa_match.group(1).strip()

    # Career goal
    goal_match = re.search(
        r"(?:goal|want|aspire|aim|plan)\s*(?:to\s+)?(?:be|become|pursue)?\s*(?:a\s+)?(\w+(?:\s+\w+)*?\s*(?:engineer|scientist|developer|researcher|specialist|expert|manager|analyst))",
        text,
        re.IGNORECASE
    )
    if goal_match:
        info["career_goal"] = goal_match.group(1).strip()

    # Budget – keep as free text: "20 lakhs", "15-20 lakhs", etc.
    budget_match = re.search(r"(\d{1,3}\s*lakhs?)", text, re.IGNORECASE)
    if budget_match:
        info["budget"] = budget_match.group(1).strip()

    # Preferred locations / countries (very simple)
    countries_match = re.findall(
        r"\b(US|USA|Canada|Germany|UK|United Kingdom|Australia|France|Singapore|Netherlands|Ireland|India)\b",
        text,
        re.IGNORECASE
    )
    if countries_match:
        cleaned = []
        for c in countries_match:
            c = c.strip()
            if not c:
                continue
            c_up = c.upper()
            if c_up == "US":
                c_norm = "USA"
            elif c_up == "UK":
                c_norm = "United Kingdom"
            else:
                c_norm = c.title()
            if c_norm not in cleaned:
                cleaned.append(c_norm)
        info["preferred_locations"] = cleaned

    # Specialization – broadened with AI/ML/Data Science, etc.
    spec_match = re.search(
        r"(?:speciali[sz]e?|specialization|field|major|focus)\s*(?:in\s+)?((?:ai|ml|machine learning|computer science|cs|data science|data analytics|software engineering|mechanical engineering|civil engineering|electronics|ece))",
        text,
        re.IGNORECASE
    )
    if spec_match:
        spec = spec_match.group(1).strip()
        lower = spec.lower()
        if lower in {"ml", "machine learning"}:
            spec = "Machine Learning"
        elif lower in {"ai"}:
            spec = "Artificial Intelligence"
        elif lower in {"cs", "computer science"}:
            spec = "Computer Science"
        elif lower in {"ece", "electronics"}:
            spec = "Electronics and Communication"
        info["specialization"] = spec

    # NEW: Academic level (high_school / undergraduate / postgraduate / working_professional)
    academic_level = _infer_academic_level_from_text(text)
    if academic_level:
        info["academic_level"] = academic_level

    # NEW: Board and Class 12 score (for high-school students)
    board = _extract_board(text)
    if board:
        info["board"] = board

    class12_score = _extract_class12_score(text)
    if class12_score:
        info["class12_score"] = class12_score

    # NEW: Competitive exams (JEE, NEET, SAT, etc.)
    exams = _extract_competitive_exams(text)
    if exams:
        info["competitive_exams"] = exams

    return info


def clean_user_pref_locations(value: Any) -> Optional[List[str]]:
    if not value:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parts = re.split(r",|\band\b", value)
        cleaned = [p.strip() for p in parts if p.strip()]
        return cleaned if cleaned else None
    return None


def build_system_instruction(mandatory_fields: List[str], optional_fields: List[str]) -> str:
    """
    Return the top-level instruction for the LLM to behave strictly and return JSON only.

    NOTE:
    - To make intake "less strict", you can simply pass an empty list for mandatory_fields
      and treat everything as optional. The model will then mostly set missing fields to null
      and avoid asking follow-up questions.
    """
    return f"""You are an academic advisor bot whose single job is to collect ONLY user-provided information.
Do NOT invent or infer missing values. If a value is missing or not provided, return it as null.
You MUST ONLY RESPOND WITH A SINGLE JSON OBJECT (no extra text) with these keys:
- {', '.join(mandatory_fields + optional_fields)}
- next_question  (string or null) — if you genuinely need one specific follow-up question to clarify something important.
- validation (object) — optional, map of field -> "ok" or "needs_clarification" if you validated the user's latest input.

Rules:
1. If you already have a value for a field (from conversation so far), put it in that key exactly as given by the user.
2. If a field is missing, set it to null. Do NOT guess.
3. Only set `next_question` if clarification is truly needed. Otherwise set it to null.
4. Do NOT include any keys other than those listed.
5. Do not include explanations, apologies, or any additional prose — JSON ONLY.
6. Keep preferred_locations as an array of country/region names or null.
7. For cgpa, class12_score, and competitive_exams, accept free-text as provided (e.g., '8.3/10', '92%', 'AIR 1500 in JEE') — do not normalize here.
"""


def validate_profile_completeness(profile: Dict[str, Any]) -> Dict[str, str]:
    """
    Soft validator for required fields.
    You can choose to ignore this entirely if you want a purely best-effort intake.

    It is now academic-level aware:
    - High-school: focuses on class12_score, board, exams, budget, preferred_locations, specialization.
    - Others: similar to original behaviour, but still optional from a system perspective.
    """
    # Default questions (for undergrad/masters)
    base_required_fields = {
        "student_name": "What's your full name?",
        "current_degree": "What degree are you currently pursuing?",
        "graduation_year": "When do you expect to graduate?",
        "cgpa": "What's your current CGPA or GPA?",
        "career_goal": "What's your main career goal?",
        "preferred_locations": "Which countries are you interested in studying?",
        "budget": "What is your approximate budget for the program?",
        "specialization": "What specialization or field would you like to focus on?"
    }

    academic_level = profile.get("academic_level")

    if academic_level == "high_school":
        required_fields = {
            "student_name": "What's your full name?",
            "board": "Which board did you study under (e.g., CBSE, ICSE, State Board)?",
            "class12_score": "What was your Class 12 / higher secondary score?",
            "competitive_exams": "Have you written any competitive exams like JEE, NEET, SAT etc.? If yes, share exam name and score/rank.",
            "preferred_locations": "Which countries are you interested in studying?",
            "budget": "What is your approximate budget for your undergraduate studies?",
            "specialization": "What field or branch (e.g., CS, Mechanical, MBBS, BBA) are you interested in?"
        }
    else:
        required_fields = base_required_fields

    missing_fields: Dict[str, str] = {}
    for field, question in required_fields.items():
        if field not in profile or profile[field] in (None, "", []):
            missing_fields[field] = question

    return missing_fields
