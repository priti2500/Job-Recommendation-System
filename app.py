import os
import json
import textwrap
from typing import List, Dict, Optional

import streamlit as st
import PyPDF2
import spacy
from spacy.matcher import PhraseMatcher
from dotenv import load_dotenv
from groq import Groq


# =========================
# Environment & Config
# =========================

# Load environment variables from .env file
load_dotenv()

# Read Groq API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client globally so it can be reused
groq_client: Optional[Groq] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def get_job_recommendation(skills: str) -> Optional[str]:
    """
    Simple example function that uses the global Groq client
    with the llama-3.1-8b-instant model to generate job recommendations
    from a plain skills string.
    """
    if not GROQ_API_KEY or client is None:
        st.error("GROQ_API_KEY is not set or Groq client is not initialized. Please check your .env file.")
        return None

    prompt = f"""
Suggest suitable job roles, required skills, missing skills, and a short career roadmap for this user:
Skills: {skills}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error while calling Groq API: {e}")
        return None


# A simple, broad list of common tech skills for matching.
# You can extend this list based on your domain.
SKILL_KEYWORDS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php", "r", "scala",
    # Data / ML
    "machine learning", "deep learning", "data analysis", "data science", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "sql", "power bi", "tableau",
    # Web / Backend
    "html", "css", "django", "flask", "fastapi", "react", "angular", "node.js", "express",
    "rest api", "graphql",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "linux",
    # Soft skills
    "communication", "leadership", "teamwork", "problem solving", "time management",
    # Other
    "excel", "project management", "agile", "scrum", "jira"
]


# =========================
# Helper Functions
# =========================

@st.cache_resource
def load_spacy_model():
    """
    Load and cache the spaCy English model.
    This avoids reloading the model on every interaction.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error(
            "spaCy model 'en_core_web_sm' is not installed.\n\n"
            "Run this command in your terminal:\n"
            "python -m spacy download en_core_web_sm"
        )
        return None


def extract_text_from_pdf(file) -> str:
    """
    Extract raw text from a PDF file using PyPDF2.
    """
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text


def preprocess_text(text: str) -> str:
    """
    Basic preprocessing: strip whitespace and collapse multiple newlines.
    This keeps it simple and beginner-friendly.
    """
    text = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def build_phrase_matcher(nlp, skill_list: List[str]) -> PhraseMatcher:
    """
    Build a spaCy PhraseMatcher for the given list of skills.
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILLS", patterns)
    return matcher


def extract_skills(nlp, text: str, skill_list: List[str]) -> List[str]:
    """
    Use spaCy and PhraseMatcher to identify skills present in the text.
    """
    if not text:
        return []

    doc = nlp(text)
    matcher = build_phrase_matcher(nlp, skill_list)
    matches = matcher(doc)

    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.lower())

    # Sort for consistent display
    return sorted(found_skills)


def build_job_recommendation_prompt(
    extracted_skills: List[str],
    user_raw_text: str
) -> str:
    """
    Build a clear instruction prompt for the LLM to generate:
    - Recommended job roles
    - Required skills per role
    - Missing skills (skill gap)
    - Career roadmap
    - Learning suggestions

    The prompt asks the model to answer in JSON for easier parsing.
    """
    skills_str = ", ".join(extracted_skills) if extracted_skills else "None"
    # Truncate long text to keep prompt size manageable
    user_summary = textwrap.shorten(user_raw_text, width=1500, placeholder=" ...")

    prompt = f"""
You are an expert career coach and job recommendation assistant.

The user's extracted skills are:
{skills_str}

The user's resume or skill description is:
\"\"\"{user_summary}\"\"\"

Based on these skills and the resume, do the following:

1. Recommend 3–5 suitable job roles.
2. For each recommended role, list:
   - A short description of the role.
   - The key required technical and soft skills.
3. Perform a skill gap analysis:
   - Which important skills are missing or weak based on the user's current skills?
4. Provide a career roadmap:
   - Clear, practical steps the user can follow over the next 6–12 months to move towards these roles.
5. Give learning suggestions:
   - Specific types of courses, certifications, or projects to work on.

Respond ONLY in valid JSON with this structure:

{{
  "recommended_jobs": [
    {{
      "role": "string",
      "description": "string",
      "required_skills": ["string", "..."]
    }}
  ],
  "missing_skills": ["string", "..."],
  "career_roadmap": ["Step 1", "Step 2", "..."],
  "learning_suggestions": ["Suggestion 1", "Suggestion 2", "..."]
}}

Do not include any text outside the JSON object.
"""
    return textwrap.dedent(prompt).strip()


def generate_job_recommendations_with_groq(
    extracted_skills: List[str],
    user_raw_text: str,
    max_tokens: int = 800,
) -> Optional[str]:
    """
    Call the Groq API (llama3-8b-8192 model) to generate job recommendations.

    - Uses the GROQ_API_KEY from the .env file.
    - Returns the model's text output or None in case of error.
    - Designed to work on Groq's free tier.
    """
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
        return None

    if groq_client is None:  
        st.error("Groq client could not be initialized. Please check your GROQ_API_KEY.")
        return None

    # Build the JSON-style instruction prompt
    prompt = build_job_recommendation_prompt(extracted_skills, user_raw_text)

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI career coach that responds in JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.7,
            max_tokens=max_tokens,
        )

        # Extract the assistant's reply text
        message = completion.choices[0].message
        return message.content if message and message.content else None

    except Exception as e:
        # Generic error handling that will also cover HTTP / auth / rate-limit errors
        error_msg = str(e)
        if "401" in error_msg:
            st.error("Groq API error 401 (Unauthorized). Please check your GROQ_API_KEY.")
        elif "403" in error_msg:
            st.error("Groq API error 403 (Forbidden). Your key may not have access to this model.")
        elif "429" in error_msg:
            st.error("Groq API error 429 (Rate limit). Please wait a bit and try again.")
        elif "500" in error_msg or "502" in error_msg or "503" in error_msg or "504" in error_msg:
            st.error("Groq API server error. Please try again later.")
        else:
            st.error(f"Unexpected error while calling Groq API: {error_msg}")

        return None


def parse_llm_json(output_text: str) -> Optional[Dict]:
    """
    Try to parse the LLM output as JSON.
    If parsing fails, return None so we can fall back to raw text display.
    """
    if not output_text:
        return None

    # Some models echo the prompt; try to extract the last JSON-looking block
    try:
        # Find the first '{' and last '}' to isolate JSON
        start = output_text.find("{")
        end = output_text.rfind("}")
        if start != -1 and end != -1 and start < end:
            json_str = output_text[start:end + 1]
        else:
            json_str = output_text

        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# =========================
# Streamlit UI
# =========================

def main():
    """
    Main function that defines the Streamlit user interface and orchestrates the workflow.
    """
    st.set_page_config(
        page_title="AI Resume-Based Job Recommender",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Generative AI Resume-Based Job Recommendation")
    st.write(
        "Upload your resume or enter your skills to get **AI-powered job recommendations**, "
        "skill gap analysis, and a personalized career roadmap."
    )

    nlp = load_spacy_model()
    if nlp is None:
        st.stop()

    # Sidebar with instructions
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.markdown(
            """
            - Upload your **PDF resume** *or* type your skills.
            - The app extracts and analyzes your skills using **spaCy**.
            - A free **Groq LLM (llama3-8b-8192)** suggests:
              - Recommended job roles  
              - Required & missing skills  
              - Career roadmap & learning ideas  
            """
        )
        st.markdown("---")
        st.markdown("**Tech stack:** Python, Streamlit, spaCy, PyPDF2, Groq API")

    # Input section
    st.subheader("1️⃣ Provide your resume or skills")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF)",
            type=["pdf"],
            help="Only PDF files are supported."
        )

    with col2:
        manual_skills = st.text_area(
            "Or manually enter your skills / experience",
            placeholder="Example: Python, SQL, data analysis, machine learning, communication...",
            height=150,
        )

    if not uploaded_file and not manual_skills.strip():
        st.info("Please upload a resume or enter your skills to proceed.")
        st.stop()

    # Process input
    st.subheader("2️⃣ Extracted skills and resume text")

    raw_text = ""

    if uploaded_file:
        st.write("**Using uploaded PDF resume.**")
        pdf_text = extract_text_from_pdf(uploaded_file)
        raw_text += pdf_text

    if manual_skills.strip():
        st.write("**Including manually entered skills/experience.**")
        raw_text += "\n" + manual_skills.strip()

    raw_text = preprocess_text(raw_text)

    with st.expander("Show extracted text (from resume and/or manual input)"):
        st.text(raw_text if raw_text else "No text extracted.")

    # Extract skills using spaCy
    extracted_skills = extract_skills(load_spacy_model(), raw_text, SKILL_KEYWORDS)

    if extracted_skills:
        st.success("Skills detected from your input:")
        st.write(", ".join(extracted_skills))
    else:
        st.warning("No predefined skills detected. The AI will still use your text as context.")

    # Button to trigger AI job recommendation
    st.subheader("3️⃣ Generate AI job recommendations")

    if st.button("Generate Recommendations"):
        if not raw_text:
            st.error("No text available to analyze. Please upload a readable PDF or enter skills.")
            st.stop()

        with st.spinner("Calling Groq AI model and generating recommendations..."):
            llm_output = generate_job_recommendations_with_groq(extracted_skills, raw_text)

        if not llm_output:
            st.error("No response from the AI model.")
            st.stop()

        # Try to parse JSON
        parsed = parse_llm_json(llm_output)

        st.subheader("4️⃣ Results")

        if parsed:
            # Structured display from JSON
            # Recommended Jobs
            recommended_jobs = parsed.get("recommended_jobs", [])
            if recommended_jobs:
                st.markdown("### 🔍 Recommended Job Roles")
                for job in recommended_jobs:
                    role = job.get("role", "Unknown role")
                    desc = job.get("description", "")
                    req_skills = job.get("required_skills", [])

                    st.markdown(f"**{role}**")
                    if desc:
                        st.write(desc)
                    if req_skills:
                        st.markdown("**Required skills:** " + ", ".join(req_skills))
                    st.markdown("---")

            # Missing Skills
            missing_skills = parsed.get("missing_skills", [])
            st.markdown("### ⚠️ Skill Gap Analysis (Missing or Weak Skills)")
            if missing_skills:
                st.write(", ".join(missing_skills))
            else:
                st.write("No significant missing skills detected, based on the model's analysis.")

            # Career Roadmap
            roadmap = parsed.get("career_roadmap", [])
            st.markdown("### 🗺️ Career Roadmap (Next 6–12 Months)")
            if roadmap:
                for i, step in enumerate(roadmap, start=1):
                    st.markdown(f"**Step {i}.** {step}")
            else:
                st.write("No roadmap provided by the model.")

            # Learning Suggestions
            suggestions = parsed.get("learning_suggestions", [])
            st.markdown("### 📚 Learning Suggestions")
            if suggestions:
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
            else:
                st.write("No specific learning suggestions provided by the model.")

            with st.expander("Show raw JSON from AI (for debugging / curiosity)"):
                st.json(parsed)
        else:
            # Fallback: show raw LLM output if JSON parsing failed
            st.warning("Could not parse structured JSON from the AI. Showing raw response instead.")
            st.text(llm_output)


if __name__ == "__main__":
    main()