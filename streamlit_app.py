import os
import PyPDF2
import docx
import io
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import streamlit as st

# Predefined skillset (expandable)
skills_list = ['python', 'machine learning', 'data science', 'nlp', 'tensorflow', 'keras', 'pandas', 'numpy', 'scikit-learn']

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 1. Resume Parsing Functions
def extract_text_from_pdf(resume_file):
    try:
        reader = PyPDF2.PdfReader(resume_file)
        return ''.join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ''

def extract_text_from_docx(resume_file):
    try:
        doc = docx.Document(io.BytesIO(resume_file.read()))
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ''

def parse_resume(resume_file):
    if resume_file.type == 'application/pdf':
        return extract_text_from_pdf(resume_file)
    elif resume_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return extract_text_from_docx(resume_file)
    else:
        print(f"Unsupported file format: {resume_file.name}")
        return ''

# 2. Preprocessing and Similarity
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(filtered_tokens)

def get_similarity(resume_text, job_desc):
    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(job_desc)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([processed_resume, processed_jd])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

# 3. Skill Extraction
def extract_skills(text):
    text = text.lower()
    return list({skill for skill in skills_list if skill in text})

# 4. Experience & Organization Extraction (Regex-based)
def extract_experience_and_education(text):
    job_titles = re.findall(r'\b(?:Data Scientist|Engineer|Developer|Manager|Intern|Analyst)\b', text, re.IGNORECASE)
    organizations = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)*\s(?:Inc|LLC|Corporation|Solutions|Technologies|Company)\b', text)
    return job_titles, organizations

# 5. Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# 6. Scoring and Ranking Resumes
def score_resumes(uploaded_resumes, job_description):
    scores = []
    
    for resume_file in uploaded_resumes:
        resume_text = parse_resume(resume_file)
        
        if resume_text.strip():
            similarity = get_similarity(resume_text, job_description)
            skills = extract_skills(resume_text)
            skills_match = len(skills) / len(skills_list) * 100

            job_titles, organizations = extract_experience_and_education(resume_text)
            experience_match = len(job_titles) / 5 * 100  # Simplified metric

            sentiment_score = analyze_sentiment(resume_text)

            total_score = (similarity + skills_match + experience_match + (sentiment_score * 100)) / 4

            scores.append({
                'Candidate': resume_file.name,
                'Match %': similarity,
                'Skills Match': skills_match,
                'Experience Match': experience_match,
                'Sentiment Score': sentiment_score * 100,
                'Total Score': round(total_score, 2)
            })
        else:
            print(f"Empty or unreadable resume: {resume_file.name}")

    df = pd.DataFrame(scores)
    df = df.sort_values(by='Total Score', ascending=False).reset_index(drop=True)
    return df

# 7. Streamlit UI
st.title("Resume Ranking System")
st.write("Upload resumes and enter the job description to rank candidates.")

# Upload resumes
uploaded_resumes = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

# Job description input
job_description = st.text_area("Enter Job Description", height=200)

# Rank Resumes Button
if st.button("Rank Resumes"):
    if not uploaded_resumes or not job_description.strip():
        st.warning("Please upload resumes and provide a job description.")
    else:
        st.info("Processing resumes...")
        ranked_df = score_resumes(uploaded_resumes, job_description)
        
        st.success("Resumes ranked successfully!")
        st.write("--- Ranked Candidates ---")
        st.dataframe(ranked_df)

        # Save to CSV
        ranked_df.to_csv("ranked_candidates.csv", index=False)
        st.write("Results saved to `ranked_candidates.csv`")
