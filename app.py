import streamlit as st

st.set_page_config(
    page_title="AI Recruitment Agent",
    page_icon="🚀",
    layout="wide"
)

import ui
from agents import ResumeAnalysisAgent
import atexit

ROLE_REQUIREMENTS = {
    "AI/ML Engineer": [
        "Python", "PyTorch", "Tensorflow", "Machine Learning", "Deep Learning", "MLOps", "Scikit-Learn", "NLP", "Computer Vision", 
        "Reinforcement Learning", "Hugging Face", "Data Engineering", "Feature Engineering", "AutoML", "Generative AI", "Agentic AI",
        "Artificial Intelligence"
    ],
    "Frontend Engineer": [
        "React", "Vue", "Angular", "HTML5", "CSS3", "JavaScript", "TypeScript", "Next.js", "Svelte", "Bootstrap", "TailWindCSS"
    ] 
}