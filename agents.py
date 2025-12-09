# agents.py
import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool


def create_agents_and_tasks(openrouter_api_key: str, serper_api_key: str):
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    serper_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()

    llm = LLM(
        model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.1,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    # =========================
    # NORMALIZER AGENT (UPDATED)
    # =========================
    normalizer_agent = Agent(
        role="Data Normalizer",
        goal="""Clean and standardize student profile data for ANY academic level.

You will receive a JSON-like profile object that MAY contain:
- academic_level (e.g., "high_school", "undergraduate", "postgraduate", "working_professional")
- student_name
- current_degree
- graduation_year
- cgpa
- board
- class12_score
- competitive_exams (list of {{ exam_name, details }})
- career_goal
- preferred_locations
- budget
- specialization
- raw_user_text (the original free-text description from the student)

Your job:
1. Do NOT invent information, but gently clean and standardize whatever is present.
2. For grading:
   - High school: keep class12_score as free text but make it clear (e.g., "92% (Class 12, CBSE)").
   - College: normalize CGPA notation to a consistent "X / Y" style where possible.
3. For competitive_exams: keep exam_name and details, maybe clarify them in plain language.
4. Standardize:
   - country names
   - approximate budget currencies (₹, $, £, €) → clearly tag currency and add "approx" when needed.
5. Clearly indicate which academic level the student belongs to (if provided or strongly implied).
6. Produce a friendly, conversational summary of the cleaned and standardized profile, highlighting:
   - Key strengths (scores, exams, experience)
   - Target level (UG vs PG)
   - Location & budget preferences.

You should NOT drop any information. If something is unclear, mention it as "not specified" instead of guessing.""",
        backstory="""You are a meticulous data wrangler with deep knowledge of academic formats,
currencies, and international grading systems. You ensure consistent, validated profiles
for downstream agents while explaining your clean-up in a warm, engaging way.""",
        tools=[],
        llm=llm,
        verbose=True
    )

    # =========================
    # MATCHER AGENT (UPDATED)
    # =========================
    matcher_agent = Agent(
        role="University Matcher",
        goal="""
Identify strong university–program matches for the student, adapting your logic based on academic_level.

INPUT:
You will receive:
- normalized_profile: a cleaned, semi-structured profile prepared by the Normalizer
- raw_user_text: the student's original message (use this when fields are missing)

BEHAVIOUR:
1. First, inspect normalized_profile.academic_level.
   - If "high_school":
       → Focus on UNDERGRADUATE / BACHELOR programs that match:
         * board, class12_score
         * competitive exams (JEE, NEET, SAT etc.)
         * intended specialization (e.g., CS, MBBS, BBA)
   - Otherwise (undergraduate/postgraduate/professional):
       → Focus on MASTERS / POSTGRADUATE or appropriate advanced programs, as in the original system.

2. In BOTH cases, prioritize results using this order:
   1) Course relevance
   2) Tuition fees / budget fit
   3) Location preference
   4) Scholarship availability
   5) Institutional reputation
   6) Career prospects after graduation
   7) Authentic student reviews

3. Use SerperDevTool and ScrapeWebsiteTool with model reasoning to gather accurate, up-to-date information.

OUTPUT:
- For each suggested program (UG or PG), include:
  - University name
  - Course title
  - Location (City, Country)
  - Degree level (e.g., B.Tech, BSc, BS, MS, M.Tech, MBA)
  - Duration of program
  - Detailed fee structure:
    * Tuition fees per year (in original currency + INR equivalent)
    * Estimated living costs (original currency + INR)
    * Total annual cost (original currency + INR)
  - Key admission requirements, adapted to level:
    * High-school case: 12th marks, board requirements, exam requirements (JEE/SAT/NEET etc.)
    * PG case: degree, CGPA, tests like GRE/GMAT/IELTS if relevant.
  - Any relevant scholarships or financial support.
  - Short reason why this program fits the student.

Provide at least 10 good options when possible.
Use a friendly, conversational tone while remaining detailed and accurate.
""",
        backstory="""
You work as a global higher-education consultant who handles both:
- students just finishing school (looking for undergraduate options), and
- students/working professionals looking for masters/postgraduate programs.

You are trusted for accurate, fit-based recommendations that respect the student's budget,
preferred locations, and academic background.
""",
        tools=[serper_tool, scrape_tool],
        llm=llm,
        verbose=True
    )

    # =========================
    # SPECIALIST AGENT (unchanged logic, but text mentions academic_level conceptually)
    # =========================
    specialist_agent = Agent(
        role="University Program Specialist",
        goal="""
From the matcher's results, evaluate and rank the top programs (usually top 5–10),
whether they are undergraduate or postgraduate.

For each recommended program, clearly present:
- Course Name
- University Name
- Degree Level (e.g., B.Tech, BS, BSc, MS, M.Tech, MBA)
- Country
- Estimated Budget / Cost Fit Evaluation
- Brief Program Description (2–3 lines)
- Pros (3–5 points)
- Cons (2–4 points)
- Career Alignment Score (1–10)

After listing all programs, provide:
- A ranked Top 5–10 list with reasoning
- A final recommendation summary explaining which program is the best fit and WHY,
  taking into account the student's academic_level (high_school vs undergrad vs working).

The tone should be friendly, conversational, and expert-level—like a senior academic advisor.
""",
        backstory="""
You are a highly experienced academic consultant specializing in global university programs,
rankings, ROI, graduate outcomes, and curriculum evaluation. You help students understand which
programs suit their goals, budget, and interests, while explaining trade-offs clearly.
""",
        tools=[serper_tool, scrape_tool],
        llm=llm,
        verbose=True
    )

    scholarship_agent = Agent(
        role="Scholarship Finder",
        goal="""Find relevant scholarships based on the student profile and top-ranked universities.
You may see both undergraduate and postgraduate cases; adapt accordingly.

Provide a helpful, conversational overview of available funding opportunities.
Include scholarship name, amount, eligibility, deadline, and official link.
NEVER output in JSON format - always use natural, conversational language.
If no scholarships are found, explain this clearly and suggest alternative funding options.""",
        backstory="""You are a funding advisor with access to global scholarship data.
You excel at surfacing accurate, recent opportunities that match the student's profile.
You communicate funding options in an encouraging, practical way using natural language.""",
        tools=[serper_tool, scrape_tool],
        llm=llm,
        verbose=True
    )

    reviews_agent = Agent(
        role="Reviews Collector",
        goal="""Gather authentic reviews for top-ranked universities/programs
(undergraduate or postgraduate).

Use at least 3 distinct sources (forums, student platforms, official review sites).
Avoid first-result bias. Synthesize multiple perspectives into a comprehensive, conversational summary with ratings.""",
        backstory="""You are a student experience researcher who specializes in finding balanced,
multi-source reviews. You avoid bias and provide a reliable picture of academic and campus life.
You share insights in an engaging, relatable way that helps students understand what to expect.""",
        tools=[serper_tool, scrape_tool],
        llm=llm,
        verbose=True
    )

        # =========================
    # Q&A AGENT (RESTORED + UPDATED CONTEXT HANDLING)
    # =========================
    qa_agent = Agent(
        role="Application Guide & Consultant",
        goal="""Help the student by answering questions using:
1) The research context (all previous agents' findings)
2) Web search if context does not include the required information

Your priorities:
- Be specific, practical, and correct
- Provide clear action steps
- Cite university websites when available (via web search tools)
- Never hallucinate or guess admissions rules""",
        backstory="""You are an expert admission and application coach. You know everything 
the system has already analyzed, and fetch missing pieces online only when needed.""",
        tools=[serper_tool, scrape_tool],
        llm=llm,
        verbose=True
    )

    

    # =========================
    # TASKS (UPDATED DESCRIPTIONS)
    # =========================

    normalize_task = Task(
        description="""You are given a student profile object under the placeholder {{profile}}.

The value of {{profile}} will typically look like:
- structured_profile: a dict with fields such as
  * academic_level
  * student_name
  * current_degree
  * graduation_year
  * cgpa
  * board
  * class12_score
  * competitive_exams
  * career_goal
  * preferred_locations
  * budget
  * specialization
- raw_user_text: the original free-text student message

Step 1: Read both structured_profile (if present) and raw_user_text.
Step 2: Clean and standardize all available information WITHOUT inventing anything.
Step 3: Produce a well-formatted summary with:
- A brief introduction
- Standardized profile details in a clear, structured format (sections for Personal, Academics, Exams, Goals, Preferences, Budget)
- Any improvements or normalizations you made
- Final cleaned profile overview explicitly mentioning the academic_level (e.g., high-school student aiming for CS undergraduate programs).

Make it visually appealing and easy to read using headings and bullet points.
""",
        expected_output="""A well-formatted, visually appealing summary of the student's cleaned and standardized profile, suitable for downstream university matching.""",
        agent=normalizer_agent
    )

    match_universities_task = Task(
        description="""
Use the normalized student profile from {{profile}} to identify suitable universities and programs.

The {{profile}} input may contain:
- normalized_profile: cleaned, standardized structured fields
- raw_user_text: original free-text description

CRITICAL:
1) First, detect academic_level from normalized_profile (or infer from raw_user_text if missing).
2) If academic_level is "high_school":
   - Focus on UNDERGRADUATE programs (B.Tech, BS, BSc, MBBS, BBA, etc.).
   - Pay attention to:
     * class12_score
     * board (CBSE/ICSE/State, etc.)
     * competitive_exams (JEE, NEET, SAT, etc.)
3) Otherwise:
   - Focus on POSTGRADUATE or appropriate advanced programs (MS, M.Tech, MBA, etc.)
   - Use CGPA, current_degree, graduation_year, etc.

Rank options according to this weightage:
1) Course relevance
2) Tuition fees and budget fit
3) Location preference
4) Scholarship availability
5) College reputation
6) Post-graduation career outcomes
7) Verified student reviews

For EACH university/program, provide:
- University name
- Course title
- Degree level (UG/PG and exact name)
- Location (City, Country)
- Duration of program
- Detailed fee structure:
  * Tuition fees per year (in original currency and INR equivalent)
  * Living costs estimate (original currency and INR)
  * Total annual cost (original currency and INR)
- Admission requirements (tailored to UG vs PG case)
- Any relevant scholarship or additional details
- One short paragraph explaining why this is a good fit given the student's background.

Ensure fee information is as accurate as possible and convert all costs to INR using current exchange rates.
Provide a comprehensive, conversational response highlighting the best matches with complete financial details.
""",
        expected_output="""A detailed overview of recommended universities and programs with explanations of why they match the student's profile, including complete fee structures in both original currency and INR.""",
        agent=matcher_agent
    )

    rank_programs_task = Task(
        description="""Take the university + program recommendations from {{profile}}.

Evaluate and rank the top 5 programs based on:
- Career alignment
- Budget fit
- Country/location preference
- Specialization depth
- Reputation
- Academic_level appropriateness (UG options for high_school, PG for others)

CRITICAL: Include ALL the detailed program information from the matcher results for each ranked program.
Do NOT summarize or shorten the financial details - include the complete fee structure exactly as provided.

IMPORTANT FORMATTING RULES:
- Use proper currency symbols: £ for GBP, $ for USD, € for EUR, ₹ for INR
- Ensure INR conversions use ₹ symbol, not the original currency symbol
- No extra spaces in markdown headers
- Use exact amounts as provided by the matcher agent

For each ranked program, structure your response exactly as follows:

**🏆 RANK #X: [University Name] - [Course Name]**

**📍 Location:** [City, Country]

**🎓 Degree:** [Degree Name and Level]

**⏱️ Duration:** [Duration]


**Fee Structure:**
- Tuition: [exact amount in original currency] (approximately ₹[exact INR equivalent])
- Living Costs: [exact amount in original currency] (approximately ₹[exact INR equivalent])
- Total Annual Cost: [exact amount in original currency] (approximately ₹[exact INR equivalent])

**How it will be:** [On-campus/Online/Hybrid delivery method]

**Career Alignment Score:** [X/10] - [Brief explanation]

**Intriguing Reasoning:**
[Detailed, engaging explanation of why this program fits perfectly, unique advantages, and career benefits]

**Pros:**
- [Key advantage 1]
- [Key advantage 2]
- [Key advantage 3]

**Cons:**
- [Potential drawback 1]
- [Potential drawback 2]

**Overall Recommendation:** [Strong recommendation with confidence level]

---

Ensure every program has complete, detailed financial information with correct currency formatting.
""",
        expected_output="""A detailed ranking of the top 5 programs with complete program information, properly formatted financial details, career alignment analysis, intriguing reasoning, and balanced pros/cons for each program.""",
        agent=specialist_agent
    )

    find_scholarships_task = Task(
        description="""Find scholarships relevant to the normalized student profile {{profile}}.

The student may be:
- A high-school graduate looking for undergraduate programs, or
- An undergraduate/graduate/working professional seeking postgraduate programs.

Provide helpful information about available funding opportunities.
Write in natural, conversational language with proper spacing and formatting - do not use JSON format.

For each scholarship, include:
- Scholarship name
- Amount (in original currency and INR equivalent)
- Eligibility criteria (UG/PG, nationality, scores, exams, etc.)
- Application deadline
- Official website or application link

Ensure the response is well-formatted with proper sentences and paragraphs.
""",
        expected_output="""A comprehensive, well-formatted overview of scholarships and funding options available to the student, with proper spacing and complete details.""",
        agent=scholarship_agent
    )

    collect_reviews_task = Task(
        description="""Gather reviews for the top-ranked universities and courses from {{profile}}.

Use at least 3 credible sources per university (forums, student platforms, official review sites).
Summarize strengths, weaknesses, student experience, and career outcomes.
Provide an engaging overview of what students are saying about these programs, highlighting any differences between undergraduate and postgraduate experiences where relevant.""",
        expected_output="""A comprehensive review summary highlighting student experiences and insights for each recommended program.""",
        agent=reviews_agent
    )

    crew = Crew(
        agents=[
            normalizer_agent,
            matcher_agent,
            specialist_agent,
            scholarship_agent,
            reviews_agent
        ],
        tasks=[
            normalize_task,
            match_universities_task,
            rank_programs_task,
            find_scholarships_task,
            collect_reviews_task
        ],
        verbose=True
    )

    return crew, qa_agent, llm  # qa_agent is still created below, reused separately


def create_profile_extractor_agent(openrouter_api_key: str):
    llm = LLM(
        model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
        temperature=0.1,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    profile_extractor_agent = Agent(
        role="Profile Information Extractor",
        goal="""Extract and structure student profile information from natural language text
for ANY academic level (high-school, undergraduate, postgraduate, working professional).

Identify all relevant details about the student's background, goals, and preferences.
Provide accurate, complete extraction with proper validation, but do NOT invent values.""",
        backstory="""You are an expert at analyzing student profiles and extracting structured information.
You carefully parse text to identify academic background, test scores, career goals, financial constraints,
and country/field preferences.""",
        tools=[],
        llm=llm,
        verbose=True
    )

    return profile_extractor_agent


def extract_profile_with_llm(profile_extractor_agent, text: str):
    task = Task(
        description=f"""Extract the following information from this student profile text:

TEXT:
{text}

Extract and return as JSON with these keys:
- student_name: Full name (string or null)
- academic_level: one of ["high_school","undergraduate","postgraduate","working_professional","other"] or null
- current_degree: Current or most recent degree (string or null)
- graduation_year: Year of graduation or expected graduation (string or null)
- board: For high-school students (CBSE, ICSE, State Board, etc.) or null
- class12_score: Class 12 / higher secondary score, keep as free text (e.g. "92%", "480/500") or null
- cgpa: CGPA/GPA with scale if mentioned (e.g. "8.3/10", "3.5/4", "75%") or null
- competitive_exams: array of objects like {{ "exam_name": "...", "details": "..." }} or null
- career_goal: Main career aspiration (string or null)
- preferred_locations: List of preferred countries/regions (array of strings or null)
- budget: Budget amount with currency in free text (e.g. "20 lakhs", "25-30 lakhs", "$30,000") or null
- specialization: Field of specialization / intended major (string or null)
- intended_degree_level: "undergraduate", "postgraduate", or null

IMPORTANT RULES:
1. Do NOT infer or guess. If the user did not say it, set it to null.
2. Keep scores, budget, and exam information in free-text form (do not normalize).
3. preferred_locations should be an array of strings or null.
4. competitive_exams should be an array of small objects, each with exam_name and details.

Return ONLY a single valid JSON object, with no extra text before or after.""",
        expected_output="""JSON object with the extracted profile fields as described above.""",
        agent=profile_extractor_agent
    )

    crew = Crew(
        agents=[profile_extractor_agent],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()
    return result


def create_qa_task(qa_agent, question: str, context: str):
    qa_task = Task(
        description=f"""
You are a helpful academic advisor answering a student's question.

QUESTION:
{question}

RESEARCH CONTEXT (Use this first!):
{context}

If information is NOT present in context:
→ Use web search to retrieve accurate, current details.

Requirements:
- Direct, correct, and student-friendly answers
- Provide steps if explaining “how to apply”
- Use official sources where possible
- Do NOT invent tuition/requirements if uncertain
""",
        expected_output="""A practical, accurate answer with any missing info retrieved via tools.""",
        agent=qa_agent,
    )

    return Crew(
        agents=[qa_agent],
        tasks=[qa_task],
        verbose=True
    )

