import os
import PyPDF2
import docx
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Predefined skillset (expandable)
skills_list = ['python', 'machine learning', 'data science', 'nlp', 'tensorflow', 'keras', 'pandas', 'numpy', 'scikit-learn']

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 1. Resume Parsing Functions
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ''.join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ''

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ''

def parse_resume(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
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
def score_resumes(resume_folder, job_description):
    scores = []

    for filename in os.listdir("C:\\Users\\Golla Krishna\\OneDrive\\Documents\\resume"):
        if not (filename.endswith('.pdf') or filename.endswith('.docx')):
            continue

        file_path = os.path.join(resume_folder, filename)
        resume_text = parse_resume(file_path)

        if resume_text.strip():
            similarity = get_similarity(resume_text, job_description)
            skills = extract_skills(resume_text)
            skills_match = len(skills) / len(skills_list) * 100

            job_titles, organizations = extract_experience_and_education(resume_text)
            experience_match = len(job_titles) / 5 * 100  # Simplified metric

            sentiment_score = analyze_sentiment(resume_text)

            total_score = (similarity + skills_match + experience_match + (sentiment_score * 100)) / 4

            scores.append({
                'Candidate': filename,
                'Match %': similarity,
                'Skills Match': skills_match,
                'Experience Match': experience_match,
                'Sentiment Score': sentiment_score * 100,
                'Total Score': round(total_score, 2)
            })
        else:
            print(f"Empty or unreadable resume: {filename}")

    df = pd.DataFrame(scores)
    df = df.sort_values(by='Total Score', ascending=False).reset_index(drop=True)
    return df

# 7. Run the Program (optional for standalone testing)
if __name__ == "__main__":
    job_description = """
    We are looking for a Data Scientist with experience in Python, Machine Learning,
    Natural Language Processing, and resume parsing tools like NLTK or TextBlob.
    """
    resume_folder = "C:\\Users\\Golla Krishna\\OneDrive\\Documents\\resume"
    print("Processing resumes...\n")
    ranked_df = score_resumes(resume_folder, job_description)

    print("\n--- Ranked Candidates ---\n")
    print(ranked_df)

    ranked_df.to_csv("ranked_candidates.csv", index=False)
    print("\nResults saved to 'ranked_candidates.csv'")
