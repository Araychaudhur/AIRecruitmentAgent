# AI Recruitment Agent 🚀

An intelligent recruitment assistant powered by Large Language Models (LLMs) to streamline resume analysis, generate interview questions, and provide resume improvement suggestions. Built with Python, Streamlit, Langchain, and OpenAI.

## Overview

The AI Recruitment Agent is a web application designed to assist recruiters and candidates by automating various aspects of the hiring process. It leverages the power of LLMs like GPT-4o through the Langchain framework to provide deep insights into resumes, match candidates with job descriptions, and prepare for interviews.

## Features ✨

* **Automated Resume Parsing:** Extracts text from PDF and TXT resume files.
* **Semantic Skill Analysis:**
    * Compares resume content against required skills from a job description or predefined roles.
    * Provides an overall match score and individual skill proficiency ratings.
    * Identifies candidate's strengths and areas for improvement.
* **Retrieval Augmented Generation (RAG):**
    * Creates a vector store (FAISS) from resume content for contextual understanding.
    * Enables interactive Q&A about the resume.
* **Weakness Analysis & Improvement Suggestions:**
    * Pinpoints weaknesses in the resume concerning target skills.
    * Offers concrete suggestions and example bullet points to enhance the resume.
* **Personalized Interview Question Generation:**
    * Generates various types of interview questions (Basic, Technical, Behavioral, Scenario, Coding) tailored to the candidate's resume and the job role.
    * Allows customization of question difficulty and quantity.
* **Resume Improvement & Rewriting:**
    * Suggests improvements across different sections like Content, Format, Skills Highlighting, etc.
    * Can generate a fully rewritten, optimized version of the resume based on identified weaknesses and target roles.
* **Interactive User Interface:**
    * User-friendly web interface built with Streamlit.
    * Visualizations for analysis results, including scores and charts.
    * Downloadable analysis reports and improved resumes.

## Tech Stack 🛠️

* **Backend:** Python
* **AI/ML:**
    * Langchain
    * OpenAI API (gpt-4o)
    * FAISS (for vector similarity search)
    * PyPDF2 (for PDF processing)
* **Frontend:** Streamlit
* **Core Libraries:** `concurrent.futures` (for parallel processing), `re` (for regex operations)

## Setup and Installation ⚙️

1.  **Clone the repository:**
    ```bash
    git clone [Your GitHub Repository Link Here]
    cd AIRecruitmentAgent
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *The `requirements.txt` includes: streamlit, langchain, langchain-openai, langchain-community, openai, pypdf2, faiss-cpu, pandas, python-dotenv, matplotlib.* [cite: 1]

4.  **Set up your OpenAI API Key:**
    * You can set it as an environment variable `OPENAI_API_KEY`.
    * Alternatively, the application will prompt for it in the sidebar.

## Usage ▶️

Run the Streamlit application:
```bash
streamlit run app.py