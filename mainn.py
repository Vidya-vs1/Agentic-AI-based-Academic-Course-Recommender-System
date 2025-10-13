#!/usr/bin/env python3
"""
smart_admissions.py

Fully conversational advisor_agent for CrewAI:
- LLM asks follow-up questions for missing mandatory fields
- Accepts optional LOR PDF upload (extracts text)
- Final output: structured JSON with ONLY user-provided fields (no hallucination)
- Sends the result to crew.kickoff(inputs={'student_query': profile_json})

Install required packages:
pip install crewai crewai-tools python-dotenv pymupdf
"""

import os
import re
import json
import time
import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# For file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

# PDF reader
import fitz  # PyMuPDF

# CrewAI imports (assumes your existing environment has crewai, crewai_tools)
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# -------------------------
# Load env
# -------------------------
load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL_NAME = os.getenv("OPENROUTER_MODEL")
BASE_URL = os.getenv("OPENROUTER_BASE_URL")

# Put API keys into env if some libs read them from os.environ
if OPEN_API_KEY:
    os.environ["OPEN_API_KEY"] = OPEN_API_KEY
if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# -------------------------
# Helpers
# -------------------------
import re
import json
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# ========== Utility Function to Extract Text from PDF ==========
def extract_lor_text(pdf_path):
    """
    Try reading LOR using pdfplumber.
    If it fails or is scanned, use OCR (pytesseract).
    """
    text = ""
    try:
        # Try normal text extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        # If empty text, fallback to OCR
        if not text.strip():
            print("🧐 No readable text found. Running OCR on scanned pages...")
            pages = convert_from_path(pdf_path)
            for page in pages:
                text += pytesseract.image_to_string(page)

    except Exception as e:
        print(f"⚠️ Error reading LOR: {e}")
        return None

    return text.strip() if text else None


# ========== Helper: Extract fields from free-form user text ==========
def extract_info_from_text(text):
    info = {}

    # Basic fields via regex or keywords
    name_match = re.search(r"\b(?:i'?m|i am|my name is)\s+([A-Z][a-z]+)", text, re.IGNORECASE)
    if name_match:
        info["student_name"] = name_match.group(1).strip()

    degree_match = re.search(r"(engineering|btech|bachelor|bachelors|bsc|b\.?e\.?|ba|bcom|bca|computer science|it|mechanical|civil)", text, re.IGNORECASE)
    if degree_match:
        info["current_degree"] = degree_match.group(1).strip()

    year_match = re.search(r"(\d{4}|next year|this year|\d{2})", text, re.IGNORECASE)
    if year_match:
        info["graduation_year"] = year_match.group(1).strip()

    cgpa_match = re.search(r"cgpa\s*(?:is|:)?\s*([\d.]+)", text, re.IGNORECASE)
    if cgpa_match:
        info["cgpa"] = cgpa_match.group(1).strip()

    goal_match = re.search(r"(?:goal|want|aspire).*?(engineer|scientist|developer|researcher)", text, re.IGNORECASE)
    if goal_match:
        info["career_goal"] = goal_match.group(1).strip()

    budget_match = re.search(r"(\d{4,6}\$?)", text)
    if budget_match:
        info["budget"] = budget_match.group(1).strip()

    countries_match = re.findall(r"(US|USA|Canada|Germany|UK|Australia|France|Singapore|Netherlands)", text, re.IGNORECASE)
    if countries_match:
        info["preferred_locations"] = list(set([c.capitalize() for c in countries_match]))

    return info


# ========== Interactive Intake Agent ==========
def intake_agent():
    print("\n============================================================")
    print("🎓 SmartAdmissions — Conversational Intake with OCR")
    print("============================================================")

    print("👋 Hi — I'm your AI admissions advisor. Tell me about yourself (education, goals, preferences).")
    user_text = input("\n🧑 You: ")

    # Extract initial info
    structured_profile = extract_info_from_text(user_text)

    # Ask follow-up questions for missing fields
    required_fields = {
        "student_name": "What's your full name?",
        "current_degree": "What degree are you currently pursuing?",
        "graduation_year": "When do you expect to graduate?",
        "cgpa": "What's your current CGPA?",
        "career_goal": "What's your main career goal?",
        "preferred_locations": "Which countries are you interested in studying?",
        "budget": "What is your approximate budget for the program?",
    }

    for field, question in required_fields.items():
        if field not in structured_profile or not structured_profile[field]:
            structured_profile[field] = input(f"🎓 Advisor: {question}\n🧑 You: ")

    # Handle specialization follow-up
    specialization = input("🎓 Advisor: What specialization or field would you like to focus on? (e.g. AI/ML, Data Science, Cybersecurity)\n🧑 You: ")
    structured_profile["specialization"] = specialization if specialization else None

    # Handle optional LOR upload
    lor_choice = input("\n📄 Would you like to upload an optional Letter of Recommendation (PDF)? (y/n): ").strip().lower()
    lor_text = None

    if lor_choice == 'y':
        lor_path = input("📎 Enter the full path to your LOR PDF: ").strip()
        lor_text = extract_lor_text(lor_path)
        if lor_text:
            print("✅ LOR successfully read and stored.")
        else:
            print("⚠️ Could not extract text from LOR.")
    else:
        lor_text = None

    structured_profile["lor_text"] = lor_text
    structured_profile["special_requirements"] = None
    structured_profile["other_info"] = None

    # Final structured output
    print("\n✅ Final structured profile (only what you provided):")
    print(json.dumps(structured_profile, indent=2, ensure_ascii=False))

    return structured_profile



def call_llm_flexible(llm: LLM, prompt: str, max_retries: int = 2) -> str:
    """
    Wrapper to call the LLM instance with a few possible method names that
    different CrewAI setups may expose. Returns raw string output.
    """
    for attempt in range(max_retries + 1):
        try:
            # Try common call signature that returns dict with 'text'
            if hasattr(llm, "create"):
                out = llm.create(prompt)
                if isinstance(out, dict) and "text" in out:
                    return out["text"]
                if isinstance(out, str):
                    return out
                # if it's dict but different shape, try to stringify
                return json.dumps(out)
            # try "generate"
            if hasattr(llm, "generate"):
                out = llm.generate(prompt)
                if isinstance(out, dict) and "text" in out:
                    return out["text"]
                if isinstance(out, str):
                    return out
                return json.dumps(out)
            # try "call" or "run"
            if hasattr(llm, "call"):
                out = llm.call(prompt)
                return out if isinstance(out, str) else json.dumps(out)
            if hasattr(llm, "run"):
                out = llm.run(prompt)
                return out if isinstance(out, str) else json.dumps(out)
            # fallback: try __call__
            if callable(llm):
                out = llm(prompt)
                return out if isinstance(out, str) else json.dumps(out)
            # nothing worked
            raise RuntimeError("No supported LLM call method found on the LLM object.")
        except Exception as e:
            # retry with backoff
            if attempt < max_retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            raise

def build_system_instruction(mandatory_fields: List[str], optional_fields: List[str]) -> str:
    """Return the top-level instruction for the LLM to behave strictly and return JSON only."""
    return f"""You are an academic advisor bot whose single job is to collect **only** user-provided information.
Do NOT invent or infer missing values. If a value is missing or not provided, return it as null.
You MUST ONLY RESPOND WITH A SINGLE JSON OBJECT (no extra text) with these keys:

- {', '.join(mandatory_fields + optional_fields)}
- next_question  (string or null) — if some mandatory/required piece is missing and you need one specific follow-up question to get it.
- validation (object) — optional, map of field -> "ok" or "needs_clarification" if you validated the user's latest input.

Rules:
1. If you already have a value for a field (from conversation so far), put it in that key exactly as given by the user.
2. If a field is missing, set it to null.
3. If you need to ask something to continue, set `next_question` to exactly one follow-up question (short, polite). Otherwise set it to null.
4. Do NOT include any keys other than those listed.
5. Do not include explanations, apologies, or any additional prose — JSON ONLY.
6. Keep preferred_locations as an array of country/region names or null.
7. For cgpa, accept free-text as provided (e.g., '8.3/10' or '75%') — do not normalize here.
"""

# -------------------------
# Create LLM and Agents
# -------------------------

# Tools
serper_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

llm = LLM(
    model=MODEL_NAME,
    temperature=0.0,
    base_url=BASE_URL,
    api_key=OPEN_API_KEY,
)

# Advisor agent - note: agent's llm is used only for reasoning; we still control the conversation loop
advisor_agent = Agent(
    role="Academic Advisor",
    goal="""Interactively collect student's profile data, ask follow-up questions only when required,
    accept an optional LOR (PDF) and extract its text, and finally produce validated JSON containing only
    information explicitly provided by the user.""",
    backstory="""Friendly intake agent. Never invent data. Ask one question at a time if needed.""",
    tools=[],
    llm=llm,
    verbose=True
)

# (Define other agents the same as your earlier script; minimal placeholders here to allow crew creation)
normalizer_agent = Agent(
    role="Data Normalizer",
    goal="""Clean and standardize student profile data: expand abbreviations (B.Tech → Bachelor of Technology), 
    normalize CGPA scales (convert to x/10), standardize currencies (₹, $, £ → ISO format), 
    normalize country names, and validate graduation years. 
    Infer useful info from 'other_info' and place in the correct fields.""",
    backstory="""You are a meticulous data wrangler with deep knowledge of academic formats, currencies, 
    and international grading systems. You ensure consistent, validated profiles for downstream agents. """,
    tools=[],
    llm=llm,
    verbose=True
)
matcher_agent = Agent(
    role="University Matcher",
    goal="""Find at least 10 relevant university + program matches that align with the student profile. 
    Consider specialization, career goals, budget, and location preference. 
    Use SerperDevTool and ScrapeWebsiteTool with LLM reasoning. 
    Output JSON only.""",
    backstory="""You are an education consultant with global reach. 
    You specialize in discovering accurate, up-to-date programs aligned to students’ requirements.""",
    tools=[serper_tool, scrape_tool],
    llm=llm,
    verbose=True
)
specialist_agent = Agent(
    role="University Specialist",
    goal="""From the matcher’s results, deeply evaluate and rank the top 5 programs. 
    Consider career alignment, budget fit, country preference, specialization depth, and reputation. 
    Provide JSON output with ranking, reasoning, and a career alignment score (1–10).""",
    backstory="""You are a senior academic consultant with deep expertise in rankings, graduate outcomes, 
    and career trajectories. You specialize in narrowing down the best-fit programs for students.""",
    tools=[serper_tool, scrape_tool],
    llm=llm,
    verbose=True
)
scholarship_agent = Agent(
    role="Scholarship Finder",
    goal="""Find relevant scholarships based on the student profile and top-ranked universities. 
    Output JSON list including: scholarship name, amount, eligibility, deadline, and official link. 
    If no scholarships found, explicitly return 'No matching scholarships found'.""",
    backstory="""You are a funding advisor with access to global scholarship data. 
    You excel at surfacing accurate, recent opportunities that match the student’s profile.""",
    tools=[serper_tool, scrape_tool],
    llm=llm,
    verbose=True
)
reviews_agent = Agent(
    role="Reviews Collector",
    goal="""Gather authentic reviews for top-ranked universities/programs. 
    Use at least 3 distinct sources (forums, student platforms, official review sites). 
    Avoid first-result bias. Synthesize multiple perspectives into a summary with rating.""",
    backstory="""You are a student experience researcher who specializes in finding balanced, 
    multi-source reviews. You avoid bias and provide a reliable picture of academic and campus life.""",
    tools=[serper_tool, scrape_tool],
    llm=llm,
    verbose=True
)
report_agent = Agent(
    role="Final Report Generator",
    goal="""Combine all results (ranked programs, scholarships, reviews, profile summary) 
    into a structured recommendation report in Markdown format.""",
    backstory="""You are an expert education consultant who creates clean, actionable reports 
    summarizing all agent findings for the student.""",
    tools=[],
    llm=llm,
    verbose=True
)

# Tasks (minimal; keep as in your pipeline)
collect_profile_task = Task(
    description="""Extract student profile fields from the free-text input: {{student_query}}. 
    If mandatory fields are missing (preferred_locations, current_degree, cgpa), 
    ask a polite follow-up question. 
    Always return valid JSON with all keys filled (use null if not available).""",
    expected_output="""{
      "student_name": "...",
      "current_degree": "...",
      "specialization": "...",
      "graduation_year": "...",
      "cgpa": "...",
      "career_goal": "...",
      "preferred_locations": ["..."],
      "budget": "...",
      "special_requirements": "...",
      "other_info": "..."
    }""",
    agent=advisor_agent,
    inputs={'student_query': '{{student_query}}'}
)

normalize_task = Task(
    description="""Normalize and clean the student profile JSON: {{profile}}.
    Expand abbreviations, normalize CGPA to /10 scale, standardize currency and country names,
    validate graduation years, and infer useful info from other_info.
    Always return a consistent JSON profile.""",
    expected_output="""Normalized JSON profile with consistent values across all fields.""",
    agent=normalizer_agent,
    inputs={'profile': '{{profile}}'}
)

match_universities_task = Task(
    description="""Using the normalized student profile from {{normalize_task.output}}, 
    find at least 10 suitable universities and programs. 
    Ensure results include course name, university, location, tuition fees, and misc details. 
    Return JSON only.""",
    expected_output="""[
      {
        "university": "...",
        "course": "...",
        "location": "...",
        "tuition_fee": "...",
        "miscellaneous": "..."
      }
    ]""",
    agent=matcher_agent,
    context=[normalize_task]
)

rank_programs_task = Task(
    description="""Take the university + program JSON list from {{match_universities_task.output}}. 
    Evaluate deeply and rank the top 5 programs. 
    Include reasoning and a career_alignment_score (1–10). 
    Return JSON only.""",
    expected_output="""[
      {
        "rank": 1,
        "university": "...",
        "course": "...",
        "location": "...",
        "tuition_fee": "...",
        "matching_reasoning": "...",
        "career_alignment_score": "..."
      }
    ]""",
    agent=specialist_agent,
    context=[match_universities_task]
)

find_scholarships_task = Task(
    description="""Find scholarships relevant to the normalized student profile ({{normalize_task.output}}) 
    and the ranked universities ({{rank_programs_task.output}}). 
    Return JSON list only. If none found, return the explicit string: "No matching scholarships found".""",
    expected_output="""[
      {
        "scholarship_name": "...",
        "amount": "...",
        "eligibility": "...",
        "deadline": "...",
        "link": "..."
      }
    ] or "No matching scholarships found" """,
    agent=scholarship_agent,
    context=[normalize_task, rank_programs_task]
)

collect_reviews_task = Task(
    description="""Gather reviews for the top-ranked universities and courses from {{rank_programs_task.output}}. 
    Use at least 3 credible sources per university (forums, student platforms, official review sites). 
    Summarize strengths, weaknesses, student experience, and career outcomes. 
    Return JSON only.""",
    expected_output="""[
      {
        "university": "...",
        "course": "...",
        "review_sources": ["...", "...", "..."],
        "review_summary": "...",
        "rating": "..."
      }
    ]""",
    agent=reviews_agent,
    context=[rank_programs_task]
)

final_report_task = Task(
    description="""Generate a comprehensive Markdown report combining:
    1. Student profile summary from {{normalize_task.output}}
    2. Top 5 ranked programs from {{rank_programs_task.output}}
    3. Scholarship opportunities from {{find_scholarships_task.output}}
    4. Reviews and feedback from {{collect_reviews_task.output}}
    5. Next steps/timeline: Use current year {{current_year}} for all deadlines and timelines. Adjust any outdated dates from sources to be current or future years. For example, if a deadline is listed as 2023, make it 2025 or appropriate future date.""",
    expected_output="""Markdown formatted final recommendation report.""",
    agent=report_agent,
    context=[normalize_task, rank_programs_task, find_scholarships_task, collect_reviews_task]
)

# -----------------------------
# Crew
# -----------------------------





# Crew (assemble)
crew = Crew(
    agents=[
        advisor_agent,
        normalizer_agent,
        matcher_agent,
        specialist_agent,
        scholarship_agent,
        reviews_agent,
        report_agent
    ],
    tasks=[
        normalize_task,              # start here instead of collect_profile_task
        match_universities_task,
        rank_programs_task,
        find_scholarships_task,
        collect_reviews_task,
        final_report_task
    ],
    verbose=True
)


# -------------------------
# Conversational intake loop (LLM-driven followups)
# -------------------------

MANDATORY_FIELDS = ["student_name", "current_degree", "specialization", "graduation_year", "cgpa",
                    "career_goal", "preferred_locations", "budget" ]
OPTIONAL_FIELDS = ["special_requirements", "other_info","lor_text"]

SYSTEM_INSTRUCTION = build_system_instruction(MANDATORY_FIELDS, OPTIONAL_FIELDS)

def clean_user_pref_locations(value: Any) -> Optional[List[str]]:
    """Try to coerce a user input for locations into a list, else return None."""
    if not value:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # split on commas or 'and'
        parts = re.split(r",|\band\b", value)
        cleaned = [p.strip() for p in parts if p.strip()]
        return cleaned if cleaned else None
    return None

def try_json_loads(s: str):
    """Attempt to parse a string as JSON, returning None if it fails."""
    try:
        return json.loads(s)
    except Exception:
        return None

def run_conversational_advisor():
    """
    Runs a conversational loop where the LLM decides the next question and validates responses.
    Returns the final structured profile dict (only user-provided values; missing -> None).
    """
    print("👋 Hi — I'm your AI admissions advisor. Tell me about yourself (education, goals, preferences).")
    # conversation history for model context (we'll include only user/assistant roles in text)
    history: List[Dict[str, str]] = []
    # profile accumulates only what user explicitly gives
    profile: Dict[str, Optional[Any]] = {k: None for k in MANDATORY_FIELDS + OPTIONAL_FIELDS}

    # seed with user's initial free-text input
    user_msg = input("\n🧑 You: ").strip()
    history.append({"role": "user", "content": user_msg})

    # Extract initial info to seed the profile
    initial_extracted = extract_info_from_text(user_msg)
    for k, v in initial_extracted.items():
        if v:
            profile[k] = v

    # MAIN LOOP: ask LLM to parse current history, produce JSON with fields + next_question
    while True:
        prompt_parts = [
            SYSTEM_INSTRUCTION,
            "\nConversation history (most recent last):",
            json.dumps(history, indent=2),
            "\nImportant: Only produce a single JSON object. Do not output anything else."
        ]
        prompt = "\n\n".join(prompt_parts)

        raw_response = call_llm_flexible(advisor_agent.llm, prompt)
        parsed = try_json_loads(raw_response)

        if not isinstance(parsed, dict):
            # if parsing failed, fallback: ask user clarifying question directly
            print("⚠️ Advisor: Sorry, I couldn't understand my own parsing. Let me ask a clarifying question.")
            next_q = "Could you briefly restate your current degree and CGPA (if any)?"
            print(f"\n🎓 Advisor: {next_q}")
            user_reply = input("🧑 You: ").strip()
            history.append({"role": "assistant", "content": next_q})
            history.append({"role": "user", "content": user_reply})
            continue

        # Accept values from parsed JSON only if non-null; store them unchanged
        for key in MANDATORY_FIELDS + OPTIONAL_FIELDS:
            if key in parsed and parsed[key] is not None:
                # preserve user's original text — do not normalize here
                profile[key] = parsed[key]

        # optional validation feedback (we won't use it here; LLM may set it)
        next_question = parsed.get("next_question", None)

        # If there is a next_question, present it to the user (LLM decided it)
        if next_question:
            # Ensure it's a single short string
            if isinstance(next_question, list):
                next_question = next_question[0] if next_question else None
            print(f"\n🎓 Advisor: {next_question}")
            user_reply = input("🧑 You: ").strip()
            # record in history and loop
            history.append({"role": "assistant", "content": next_question})
            history.append({"role": "user", "content": user_reply})
            continue

        # No next_question: intake is complete (per LLM)
        break

    # After mandatory intake, offer LOR upload if not already present
    if not profile.get("lor_text"):
        # Ask user if they want to upload LOR
        lor_upload_choice = input("\n📄 Would you like to upload an optional Letter of Recommendation (PDF)? (y/n): ").strip().lower()
        if lor_upload_choice == "y":
            lor_path = input("Enter the full path to the PDF file: ").strip().strip('"').strip("'")
            if lor_path and os.path.exists(lor_path):
                lor_text = extract_lor_text(lor_path)
                if lor_text:
                    profile["lor_text"] = lor_text
                    print("✅ LOR processed and attached to your profile.")
                else:
                    print("⚠️ LOR could not be read or was empty. Skipping.")
            else:
                print("⚠️ File not found; skipping LOR.")

    # Post-process preferred_locations: try to convert strings -> list
    if profile.get("preferred_locations") and not isinstance(profile["preferred_locations"], list):
        cleaned = clean_user_pref_locations(profile["preferred_locations"])
        profile["preferred_locations"] = cleaned

    # Ensure final JSON contains only the allowed keys and explicit values (or null)
    final_profile = {k: (profile.get(k) if profile.get(k) is not None else None) for k in MANDATORY_FIELDS + OPTIONAL_FIELDS}

    print("\n✅ Final structured profile (only what you provided):")
    print(json.dumps(final_profile, indent=2, ensure_ascii=False))
    return final_profile

# -------------------------
# Main: run advisor then kickoff Crew
# -------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 SmartAdmissions — Conversational Intake")
    print("=" * 60)

    # Step 1: Run interactive conversational advisor
    structured_profile = run_conversational_advisor()

    print("\n✅ Final structured profile (only what you provided):")
    print(json.dumps(structured_profile, indent=2))

    print("\n🟢 Passing collected profile to Crew pipeline...")

    # Step 3: Run the rest of the Crew (skipping first advisor task)
    current_year = datetime.datetime.now().year
    result = crew.kickoff(inputs={'profile': structured_profile, 'current_year': current_year})

    print("\n" + "=" * 60)
    print("📋 FINAL RECOMMENDATION REPORT")
    print("=" * 60)
    print(result)

