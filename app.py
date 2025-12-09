import streamlit as st
import json
import os
import tempfile

from utils import extract_lor_text, extract_info_from_text
from agents import (
    create_agents_and_tasks,
    create_qa_task,
    create_profile_extractor_agent,
    extract_profile_with_llm,
)

st.set_page_config(
    page_title="SmartAdmissions - AI Course Recommender",
    page_icon="🎓",
    layout="wide",
)


def init_session_state():
    if "api_keys_set" not in st.session_state:
        st.session_state.api_keys_set = False
    if "openrouter_api_key" not in st.session_state:
        st.session_state.openrouter_api_key = ""
    if "serper_api_key" not in st.session_state:
        st.session_state.serper_api_key = ""
    if "profile_complete" not in st.session_state:
        st.session_state.profile_complete = False
    if "student_profile" not in st.session_state:
        st.session_state.student_profile = {}
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = 0
    if "agent_results" not in st.session_state:
        st.session_state.agent_results = {
            "normalized_profile": None,
            "matched_programs": None,
            "ranked_programs": None,
            "scholarships": None,
            "reviews": None,
        }


def render_api_key_section():
    st.title("🎓 SmartAdmissions - AI Course Recommender")
    st.markdown("### Welcome to your AI-powered university and course recommendation system!")

    st.markdown("---")
    st.markdown("### 🔑 API Key Configuration")
    st.markdown("To get started, you'll need API keys from the following services:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### OpenRouter API Key")
        st.markdown("Get your free API key from [OpenRouter](https://openrouter.ai/keys)")
        st.markdown("*This powers our AI agents with Meta Llama 3.3*")
        openrouter_key = st.text_input(
            "Enter OpenRouter API Key",
            type="password",
            value=st.session_state.openrouter_api_key,
            key="openrouter_input",
        )

    with col2:
        st.markdown("#### Serper API Key")
        st.markdown("Get your API key from [Serper.dev](https://serper.dev/api-key)")
        st.markdown("*This enables web search capabilities for our agents*")
        serper_key = st.text_input(
            "Enter Serper API Key",
            type="password",
            value=st.session_state.serper_api_key,
            key="serper_input",
        )

    st.markdown("---")

    if st.button("✅ Save API Keys and Continue", type="primary", use_container_width=True):
        if openrouter_key and serper_key:
            st.session_state.openrouter_api_key = openrouter_key
            st.session_state.serper_api_key = serper_key
            st.session_state.api_keys_set = True
            st.success("API Keys saved successfully! You can now proceed with your profile.")
            st.rerun()
        else:
            st.error("Please provide both API keys to continue.")


def render_profile_intake():
    st.title("🎓 Tell Us About Yourself")
    st.markdown(
        "Share your educational background, goals, and preferences in your own words. "
        "Our AI will extract the information!"
    )

    st.markdown("---")

    with st.form("initial_profile_form"):
        st.markdown("### 📝 Tell us about yourself")

        initial_text = st.text_area(
            "Describe your background, goals, and preferences",
            placeholder=(
                "Example: I'm John Smith, currently in 12th (CBSE) with 92%. I wrote JEE Main and "
                "scored 95 percentile. I want to do a B.Tech in Computer Science. "
                "I'm interested in colleges in India and Germany, and my budget is around 20 lakhs.\n\n"
                "Or: I'm John Smith, currently pursuing B.Tech in Computer Science. I'll graduate in 2025 with a CGPA of 8.5. "
                "I want to become an AI/ML engineer and am interested in studying in the US or Canada for a Master's. "
                "My budget is around $30,000 per year and I'm particularly interested in machine learning specialization."
            ),
            height=200,
            help=(
                "Share everything naturally — your name, current class or degree, marks/CGPA, "
                "competitive exams (JEE/NEET/SAT/etc.), career goals, preferred countries, "
                "budget, and areas of interest."
            ),
        )

        st.markdown("### 📄 Optional: Letter of Recommendation")
        lor_file = st.file_uploader(
            "Upload LOR PDF (Optional)",
            type=["pdf"],
            help="Upload a Letter of Recommendation. We'll extract the text using OCR if needed.",
        )

        submitted = st.form_submit_button("🔍 Analyze My Profile", type="primary", use_container_width=True)

        if submitted:
            if not initial_text.strip():
                st.error("Please tell us about yourself in the text area above.")
            else:
                # 1) Basic regex / NLP extraction
                profile = extract_info_from_text(initial_text)

                # 2) LOR text extraction (optional)
                if lor_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(lor_file.read())
                        tmp_path = tmp_file.name

                    lor_text = extract_lor_text(tmp_path)
                    if lor_text:
                        profile["lor_text"] = lor_text
                        st.success("✅ LOR successfully extracted!")
                    else:
                        st.warning("⚠️ Could not extract text from LOR. Continuing without it.")

                    os.unlink(tmp_path)

                # 3) LLM-based profile extraction (hybrid mode)
                try:
                    extractor_agent = create_profile_extractor_agent(
                        st.session_state.openrouter_api_key
                    )
                    llm_result = extract_profile_with_llm(extractor_agent, initial_text)

                    # parse using the same helper as other agents
                    llm_profile = parse_agent_output(llm_result)

                    # Try to interpret LLM output as JSON dict if it's a string
                    if isinstance(llm_profile, str):
                        try:
                            parsed = json.loads(llm_profile)
                            if isinstance(parsed, dict):
                                llm_profile = parsed
                        except Exception:
                            # leave as string if not JSON
                            llm_profile = None

                    if isinstance(llm_profile, dict):
                        # merge LLM fields into regex-based profile (LLM overrides only when it has a value)
                        for k, v in llm_profile.items():
                            if v not in (None, "", []):
                                profile[k] = v

                except Exception as e:
                    st.warning(f"LLM-based profile extraction failed, using regex-only. Error: {e}")

                # 4) Always keep the raw user input for downstream reasoning
                profile["raw_input_text"] = initial_text

                st.session_state.student_profile = profile
                st.session_state.profile_complete = True
                st.success("✅ Profile received! Processing your recommendations...")
                st.rerun()


def render_agent_execution():
    st.title("🤖 AI Agents Working on Your Recommendations")
    st.markdown(
        "Our specialized AI agents are analyzing your profile and searching for the best matches..."
    )

    progress_container = st.container()

    with progress_container:
        with st.spinner("Initializing AI agents..."):
            try:
                crew, qa_agent, llm = create_agents_and_tasks(
                    st.session_state.openrouter_api_key,
                    st.session_state.serper_api_key,
                )

                st.session_state.qa_agent = qa_agent
                st.session_state.llm = llm

                profile_json = json.dumps(st.session_state.student_profile, indent=2)

                st.info("🔄 Agent Pipeline Started...")

                agent_statuses = st.empty()

                # Initialize agent results in session state
                if "agent_results" not in st.session_state:
                    st.session_state.agent_results = {
                        "normalized_profile": None,
                        "matched_programs": None,
                        "ranked_programs": None,
                        "scholarships": None,
                        "reviews": None,
                    }

                current_agent = st.session_state.get("current_agent", 0)

                # Normalizer Agent
                if current_agent >= 0 and st.session_state.agent_results["normalized_profile"] is None:
                    with agent_statuses.container():
                        st.markdown("#### Agent Progress:")
                        st.markdown("🔄 **Normalizer Agent** - Cleaning and standardizing your profile...")
                        st.markdown("⏳ **Course Matcher Agent** - Waiting...")
                        st.markdown("⏳ **Course Specialist Agent** - Waiting...")
                        st.markdown("⏳ **Scholarship Agent** - Waiting...")
                        st.markdown("⏳ **Reviews Agent** - Waiting...")

                    try:
                        from crewai import Crew

                        temp_crew = Crew(
                            agents=[crew.tasks[0].agent],
                            tasks=[crew.tasks[0]],
                            verbose=True,
                        )
                        normalize_result = temp_crew.kickoff(inputs={"profile": profile_json})
                        st.session_state.agent_results["normalized_profile"] = parse_agent_output(
                            normalize_result
                        )
                        st.session_state.current_agent = 1
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error in normalizer agent: {type(e).__name__}: {str(e)}"
                        if hasattr(e, "__cause__") and e.__cause__:
                            error_msg += (
                                f"\nCaused by: {type(e.__cause__).__name__}: {str(e.__cause__)}"
                            )
                        st.error(error_msg)
                        st.session_state.processing = False
                        return

                # Course Matcher Agent
                if current_agent >= 1 and st.session_state.agent_results["matched_programs"] is None:
                    with agent_statuses.container():
                        st.markdown("#### Agent Progress:")
                        st.markdown("✅ **Normalizer Agent** - Complete")
                        st.markdown("🔄 **Course Matcher Agent** - Finding suitable universities...")
                        st.markdown("⏳ **Course Specialist Agent** - Waiting...")
                        st.markdown("⏳ **Scholarship Agent** - Waiting...")
                        st.markdown("⏳ **Reviews Agent** - Waiting...")

                    try:
                        from crewai import Crew

                        temp_crew = Crew(
                            agents=[crew.tasks[1].agent],
                            tasks=[crew.tasks[1]],
                            verbose=True,
                        )
                        match_result = temp_crew.kickoff(
                            inputs={
                                "profile": st.session_state.agent_results["normalized_profile"]
                            }
                        )
                        st.session_state.agent_results["matched_programs"] = parse_agent_output(
                            match_result
                        )
                        st.session_state.current_agent = 2
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error in matcher agent: {str(e)}")
                        st.session_state.processing = False
                        return

                # Course Specialist Agent
                if current_agent >= 2 and st.session_state.agent_results["ranked_programs"] is None:
                    with agent_statuses.container():
                        st.markdown("#### Agent Progress:")
                        st.markdown("✅ **Normalizer Agent** - Complete")
                        st.markdown("✅ **Course Matcher Agent** - Complete")
                        st.markdown("🔄 **Course Specialist Agent** - Ranking programs...")
                        st.markdown("⏳ **Scholarship Agent** - Waiting...")
                        st.markdown("⏳ **Reviews Agent** - Waiting...")

                    try:
                        from crewai import Crew

                        temp_crew = Crew(
                            agents=[crew.tasks[2].agent],
                            tasks=[crew.tasks[2]],
                            verbose=True,
                        )
                        rank_result = temp_crew.kickoff(
                            inputs={
                                "profile": st.session_state.agent_results["matched_programs"]
                            }
                        )
                        st.session_state.agent_results["ranked_programs"] = parse_agent_output(
                            rank_result
                        )
                        st.session_state.current_agent = 3
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error in specialist agent: {str(e)}")
                        st.session_state.processing = False
                        return

                # Scholarship Agent
                if current_agent >= 3 and st.session_state.agent_results["scholarships"] is None:
                    with agent_statuses.container():
                        st.markdown("#### Agent Progress:")
                        st.markdown("✅ **Normalizer Agent** - Complete")
                        st.markdown("✅ **Course Matcher Agent** - Complete")
                        st.markdown("✅ **Course Specialist Agent** - Complete")
                        st.markdown(
                            "🔄 **Scholarship Agent** - Finding funding opportunities..."
                        )
                        st.markdown("⏳ **Reviews Agent** - Waiting...")

                    try:
                        from crewai import Crew

                        temp_crew = Crew(
                            agents=[crew.tasks[3].agent],
                            tasks=[crew.tasks[3]],
                            verbose=True,
                        )
                        scholarship_result = temp_crew.kickoff(
                            inputs={
                                "profile": st.session_state.agent_results["normalized_profile"]
                            }
                        )
                        st.session_state.agent_results["scholarships"] = parse_agent_output(
                            scholarship_result
                        )
                        st.session_state.current_agent = 4
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error in scholarship agent: {str(e)}")
                        st.session_state.processing = False
                        return

                # Reviews Agent
                if current_agent >= 4 and st.session_state.agent_results["reviews"] is None:
                    with agent_statuses.container():
                        st.markdown("#### Agent Progress:")
                        st.markdown("✅ **Normalizer Agent** - Complete")
                        st.markdown("✅ **Course Matcher Agent** - Complete")
                        st.markdown("✅ **Course Specialist Agent** - Complete")
                        st.markdown("✅ **Scholarship Agent** - Complete")
                        st.markdown("🔄 **Reviews Agent** - Gathering student feedback...")

                    try:
                        from crewai import Crew

                        temp_crew = Crew(
                            agents=[crew.tasks[4].agent],
                            tasks=[crew.tasks[4]],
                            verbose=True,
                        )
                        review_result = temp_crew.kickoff(
                            inputs={
                                "profile": st.session_state.agent_results["ranked_programs"]
                            }
                        )
                        st.session_state.agent_results["reviews"] = parse_agent_output(
                            review_result
                        )
                        st.session_state.current_agent = 5
                        st.session_state.processing = False
                        st.success("✅ All agents completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error in reviews agent: {str(e)}")
                        st.session_state.processing = False
                        return

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.session_state.processing = False
                if st.button("🔄 Try Again"):
                    st.rerun()


def rating_to_stars(rating):
    try:
        rating = float(rating)
        full_stars = int(rating)
        half_star = 1 if rating - full_stars >= 0.5 else 0
        empty_stars = 5 - full_stars - half_star
        stars = "⭐" * full_stars + "⭐" * half_star + "☆" * empty_stars
        return f"{rating}/5 {stars}"
    except Exception:
        return str(rating)


def parse_agent_output(task_output):
    try:
        import re

        # crewai's TaskOutput sometimes has .json or .raw
        if hasattr(task_output, "json") and task_output.json:
            return task_output.json
        if hasattr(task_output, "raw") and task_output.raw:
            output_text = task_output.raw
        else:
            output_text = str(task_output)

        if "No matching scholarships found" in output_text:
            return "No matching scholarships found"

        # Try JSON array
        json_array_match = re.search(r"\[.*\]", output_text, re.DOTALL)
        if json_array_match:
            return json.loads(json_array_match.group())

        # Try JSON object
        json_dict_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if json_dict_match:
            return json.loads(json_dict_match.group())

        # If no JSON found, return the text as natural language
        return output_text.strip()
    except Exception:
        # Return the raw text if JSON parsing fails
        return str(task_output).strip()


def render_results():
    st.title("🎯 Your Personalized Course Recommendations")

    normalized_profile = st.session_state.agent_results.get("normalized_profile")
    matched_programs = st.session_state.agent_results.get("matched_programs")
    ranked_programs = st.session_state.agent_results.get("ranked_programs")
    scholarships = st.session_state.agent_results.get("scholarships")
    reviews = st.session_state.agent_results.get("reviews")

    st.markdown("---")

    tabs = st.tabs(
        [
            "👤 Profile Summary",
            "📊 Top Programs",
            "💰 Scholarships",
            "⭐ Reviews",
            "📋 All Results",
            "💬 Ask Questions",
        ]
    )

    with tabs[0]:
        st.markdown("### 👤 Your Normalized Profile")
        st.markdown("Here's your cleaned and standardized profile that was used for recommendations:")

        if normalized_profile:
            st.markdown(normalized_profile)
        else:
            st.info("🔄 Normalizer Agent is still processing your profile...")
            st.markdown("Please wait while we clean and standardize your information.")

    with tabs[1]:
        st.markdown("### 🏆 Top Recommended Programs")

        if ranked_programs:
            import re
        # Split blocks by RANK heading (supports variations like "**RANK #1" or "RANK #1")
            programs = re.split(r"\*\* ?RANK", str(ranked_programs))

            for block in programs:
                block = block.strip()
                if not block:
                    continue

                formatted = "**RANK " + block  # Restore removed text

            # 🔥 Fix only heading spacing (not bullet lists!)
                formatted = formatted.replace("**RANK", "\n\n**🏆 RANK")
                formatted = formatted.replace("** Location:", "\n\n**📍 Location:")
                formatted = formatted.replace("** Degree:", "\n\n**🎓 Degree:")
                formatted = formatted.replace("** Duration:", "\n\n**⏱️ Duration:")
                formatted = formatted.replace("** Fee Structure:", "\n\n**💰 Fee Structure:")
                formatted = formatted.replace("** How it will be:", "\n\n**📚 Delivery:")
                formatted = formatted.replace("** Career Alignment Score:", "\n\n**⭐ Career Alignment Score:")
                formatted = formatted.replace("** Intriguing Reasoning:", "\n\n**💡 Why This Program Fits:")
                formatted = formatted.replace("** Pros:", "\n\n**✅ Pros:**\n")
                formatted = formatted.replace("** Cons:", "\n\n**⚠️ Cons:**\n")
                

            # Cleanup any accidental triple spacing
                formatted = formatted.replace("\n\n\n", "\n\n")

            # UI card divider
                st.markdown("---")
                st.markdown(formatted, unsafe_allow_html=True)

        else:
            st.info("🔄 Course Specialist Agent is still ranking programs...")
            st.markdown(
            """
            The Course Specialist Agent analyzes your profile and evaluates programs 
            based on **career alignment**, **budget**, **country preferences**, 
            and **program reputation**.  
            Please wait while it finalizes the rankings.
            """
        )


    with tabs[2]:
        st.markdown("### 💰 Scholarship Opportunities")
        st.markdown("Relevant scholarships based on your profile and selected programs:")

        if scholarships:
            st.markdown(scholarships)
        else:
            st.info("🔄 Scholarship Agent is still finding funding opportunities...")
            st.markdown(
                "This agent searches for scholarships, grants, and financial aid options that match your profile and target universities."
            )

    with tabs[3]:
        st.markdown("### ⭐ University & Program Reviews")
        st.markdown("What students say about these programs:")

        if reviews:
            st.markdown(reviews)
        else:
            st.info("🔄 Reviews Agent is still gathering student feedback...")
            st.markdown(
                "This agent collects authentic reviews from multiple sources to give you insights into the student experience at these universities."
            )

    with tabs[4]:
        st.markdown("### 📋 Complete Analysis Results")
        st.markdown("Full output from all AI agents:")

        agent_names = [
            "Normalizer",
            "Course Matcher",
            "Course Specialist",
            "Scholarship Finder",
            "Reviews Collector",
        ]
        agent_keys = [
            "normalized_profile",
            "matched_programs",
            "ranked_programs",
            "scholarships",
            "reviews",
        ]

        for agent_name, key in zip(agent_names, agent_keys):
            result_text = st.session_state.agent_results.get(key)
            with st.expander(f"**{agent_name} Output**"):
                if result_text:
                    st.markdown(result_text)
                else:
                    st.info(f"🔄 {agent_name} is still processing...")

    with tabs[5]:
        st.markdown("### 💬 Ask Follow-up Questions")
        st.markdown("Have questions about the recommendations? Ask our Q&A agent!")

        for qa in st.session_state.qa_history:
            with st.chat_message("user"):
                st.write(qa['question'])
            with st.chat_message("assistant"):
                st.write(qa['answer'])

        question = st.chat_input("Ask a question about your recommendations...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        context = str(st.session_state.agent_results)
                        qa_crew = create_qa_task(
                            st.session_state.qa_agent,
                            question,
                            context
                        )
                        answer = qa_crew.kickoff()
                        answer_text = str(answer)

                        st.write(answer_text)

                        st.session_state.qa_history.append({
                            'question': question,
                            'answer': answer_text
                        })

                    except Exception as e:
                        st.error(f"Error answering question: {str(e)}")

    st.markdown("---")

    # Feedback and refinement section
    st.markdown("### 💭 Not satisfied with the recommendations?")
    st.markdown("If you'd like us to refine the suggestions based on specific preferences or concerns, let us know!")

    feedback = st.text_area(
        "Tell us what you'd like to change (e.g., 'I prefer programs in Canada only' or 'Show me more affordable options')",
        key="refinement_feedback"
    )

    if st.button("🔄 Refine Recommendations", type="secondary", use_container_width=True):
        if feedback.strip():
            # Update profile with feedback
            st.session_state.student_profile['user_feedback'] = feedback.strip()
            # Reset agent results to re-run
            st.session_state.agent_results = {
                'normalized_profile': None,
                'matched_programs': None,
                'ranked_programs': None,
                'scholarships': None,
                'reviews': None
            }
            st.success("✅ Refining recommendations based on your feedback...")
            st.rerun()
        else:
            st.warning("Please provide feedback to refine the recommendations.")

    if st.button("🔄 Start New Search", type="primary"):
        st.session_state.profile_complete = False
        st.session_state.student_profile = {}
        st.session_state.qa_history = []
        st.session_state.current_agent = 0
        st.session_state.agent_results = {
            "normalized_profile": None,
            "matched_programs": None,
            "ranked_programs": None,
            "scholarships": None,
            "reviews": None,
        }
        st.rerun()


def main():
    init_session_state()

    if not st.session_state.api_keys_set:
        render_api_key_section()
    elif not st.session_state.profile_complete:
        render_profile_intake()
    elif st.session_state.current_agent < 5:
        if not st.session_state.processing:
            st.session_state.processing = True
        render_agent_execution()
    else:
        render_results()


if __name__ == "__main__":
    main()
