import streamlit as st
import pdfplumber
import nltk
import spacy
import matplotlib.pyplot as plt
from skills import skills_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Setup
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += str(page.extract_text())
    return text.lower()

# -------------------------------
# Extract skills
# -------------------------------
def extract_skills(text):
    return [skill for skill in skills_list if skill in text]

# -------------------------------
# Calculate score
# -------------------------------
def calculate_score(skills):
    return min(len(skills) * 10, 100)

# -------------------------------
# Suggestions
# -------------------------------
def suggestions(skills, text):
    sug = []
    if len(skills) < 5:
        sug.append("Add more technical skills")
    if "project" not in text:
        sug.append("Add project section")
    if "internship" not in text:
        sug.append("Add internship experience")
    if "objective" not in text:
        sug.append("Add career objective")
    return sug

# -------------------------------
# Section Detection
# -------------------------------
def detect_sections(text):
    sections = ["education", "projects", "skills", "experience", "certifications"]
    return [sec for sec in sections if sec in text]

# -------------------------------
# Job Match (ATS)
# -------------------------------
def match_job(resume, jd):
    cv = CountVectorizer().fit_transform([resume, jd])
    similarity = cosine_similarity(cv)[0][1]
    return round(similarity * 100, 2)

# -------------------------------
# AI Suggestion
# -------------------------------
def ai_suggestion(score):
    if score > 80:
        return "Your resume is strong. Apply confidently to top companies."
    elif score > 50:
        return "Improve your projects and add more relevant skills."
    else:
        return "Add skills, projects, and internships to improve your resume."

# -------------------------------
# Generate PDF Report
# -------------------------------
def generate_pdf(score, skills, sug):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Resume Score: {score}", styles['Normal']))
    content.append(Paragraph(f"Skills: {', '.join(skills)}", styles['Normal']))

    for s in sug:
        content.append(Paragraph(f"- {s}", styles['Normal']))

    doc.build(content)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🚀 Smart Resume Analyzer")

file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd = st.text_area("Paste Job Description (Optional)")

if file:
    text = extract_text(file)

    st.subheader("📄 Extracted Text")
    st.write(text[:1000])

    skills = extract_skills(text)
    score = calculate_score(skills)
    sug = suggestions(skills, text)
    sections = detect_sections(text)
    ai_msg = ai_suggestion(score)

    # Skills
    st.subheader("🧠 Skills Found")
    st.write(skills)

    # Score
    st.subheader("📊 Resume Score")
    st.write(f"{score}/100")

    # Graph
    st.subheader("📈 Skill Graph")
    if skills:
        plt.bar(skills, [1]*len(skills))
        st.pyplot(plt)
    else:
        st.write("No skills detected")

    # Sections
    st.subheader("📂 Sections Found")
    st.write(sections)

    # Suggestions
    st.subheader("💡 Suggestions")
    for s in sug:
        st.write("-", s)

    # AI Feedback
    st.subheader("🤖 AI Feedback")
    st.write(ai_msg)

    # Job Match
    if jd:
        match = match_job(text, jd.lower())
        st.subheader("🎯 Job Match")
        st.write(f"{match}%")

    # PDF Download
    if st.button("Generate PDF Report"):
        generate_pdf(score, skills, sug)
        with open("report.pdf", "rb") as f:
            st.download_button("📥 Download Report", f, file_name="resume_report.pdf")