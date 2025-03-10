'''import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load Datasets
job_df = pd.read_csv("AI_Skill_Profiling_Dataset_100k_with_Future_Learning_Resources.csv")
user_df = pd.read_csv("cse_skills_dataset_200k.csv")
company_df = pd.read_csv("company_dataset.csv")

# Standardize column names
user_df.columns = user_df.columns.str.strip().str.lower()
company_df.columns = company_df.columns.str.strip().str.lower()

# Ensure lowercase for skills and locations
user_df['skills'] = user_df['skills'].astype(str).str.lower().str.strip()
user_df['location'] = user_df['location'].astype(str).str.lower().str.strip()

# Convert experience column to numeric
user_df['experience (years)'] = pd.to_numeric(user_df['experience (years)'], errors='coerce').fillna(0)

# TF-IDF Vectorizer for Job Prediction
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(job_df["Skills"]).toarray()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(job_df["Suggested Job"])
job_model = RandomForestClassifier(n_estimators=100, random_state=42)
job_model.fit(X, y)

# Initialize GPT-2 for Resume Generation
@st.cache_resource
def load_gpt_model():
    try:
        return pipeline('text-generation', model='gpt2')
    except Exception as e:
        st.error(f"Error loading GPT-2: {e}")
        return None

generator = load_gpt_model()

# TF-IDF Vectorizer for Talent Matching
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(", "), lowercase=True)
user_skill_matrix = vectorizer.fit_transform(user_df['skills'])

def predict_job(skills):
    """Predicts a job based on skills using trained model"""
    skills_tfidf = tfidf.transform([" ".join(skills)])
    if skills_tfidf.nnz == 0:
        return "No matching job found."
    
    prediction = job_model.predict(skills_tfidf)
    return label_encoder.inverse_transform(prediction)[0]

def generate_resume(form_data):
    """Generates a career objective using GPT-2"""
    if not generator:
        return "Career objective generation is unavailable."

    skills_text = ', '.join(form_data.get("skills", "").split(","))
    education = form_data.get("education", "a strong educational background")
    job_description = form_data.get("jobDescription", "varied professional experience")
    projects = form_data.get("projects", "several impactful projects")
    certifications = form_data.get("certifications", "relevant certifications")

    prompt = (
        f"I am passionate about my work and have a deep understanding of {skills_text}. "
        f"My education in {education} and experience in {job_description} have shaped my journey. "
        f"I have successfully completed projects such as {projects} and hold certifications in {certifications}. "
        f"I am excited about contributing my expertise."
    )

    try:
        generated = generator(prompt, max_length=100, num_return_sequences=1)
        return generated[0]['generated_text'].strip() if generated else "Unable to generate a career objective."
    except Exception as e:
        return f"Error: {e}"

def match_users_to_company(company_skills, company_experience, company_location, top_n=5):
    """Finds the best candidates for a given job"""
    company_skills_vector = vectorizer.transform([company_skills.lower()])
    similarity_scores = cosine_similarity(company_skills_vector, user_skill_matrix).flatten()

    # Apply experience and location weightage
    experience_weight = 0.3  
    experience_match = (user_df['experience (years)'] >= company_experience).astype(float) * experience_weight
    location_weight = 0.2  
    location_match = (user_df['location'] == company_location.lower()).astype(float) * location_weight

    final_scores = similarity_scores + experience_match + location_match
    user_indices = final_scores.argsort()[::-1][:top_n]

    recommended_users = user_df.iloc[user_indices].copy()
    recommended_users["Score"] = final_scores[user_indices]

    return recommended_users

# Streamlit UI
st.title("🔍 AI-Powered Job & Talent Matcher")

# Sidebar Navigation
menu = st.sidebar.radio("Choose an option:", ["Job Prediction", "Resume Generator", "Talent Matcher"])

if menu == "Job Prediction":
    st.header("💼 Predict Your Ideal Job")
    user_skills = st.text_area("Enter your skills (comma-separated):").strip()

    if st.button("Predict Job"):
        if user_skills:
            skills_list = [skill.strip() for skill in user_skills.split(",") if skill.strip()]
            predicted_job = predict_job(skills_list)
            st.success(f"🎯 Suggested Job: **{predicted_job}**")
        else:
            st.warning("⚠️ Please enter your skills.")

elif menu == "Resume Generator":
    st.header("📄 Generate Your Resume")
    
    name = st.text_input("Name:")
    phone = st.text_input("Phone Number:")
    email = st.text_input("Email:")
    skills = st.text_area("Skills (comma-separated):")
    education = st.text_area("Education:")
    job_desc = st.text_area("Job Experience:")
    projects = st.text_area("Projects:")
    certifications = st.text_area("Certifications:")

    if st.button("Generate Resume"):
        form_data = {
            "name": name, "phoneNumber": phone, "email": email,
            "skills": skills, "education": education,
            "jobDescription": job_desc, "projects": projects, "certifications": certifications
        }
        career_obj = generate_resume(form_data)

        st.subheader("📜 Generated Resume")
        st.text(f"**Name:** {name}")
        st.text(f"**Contact:** {phone} | **Email:** {email}")
        st.markdown(f"**Career Objective:** {career_obj}")
        st.text(f"**Skills:** {skills}")
        st.text(f"**Education:** {education}")
        st.text(f"**Projects:** {projects}")
        st.text(f"**Certifications:** {certifications}")

elif menu == "Talent Matcher":
    st.header("🔍 Find the Best Candidates for Your Job")

    # Company selection
    company_options = company_df['company'].tolist()
    selected_company = st.selectbox("🏢 Select a company", company_options)

    if st.button("Find Candidates"):
        company_row = company_df[company_df['company'] == selected_company].iloc[0]
        required_skills = company_row['required skills']
        min_experience = company_row['min experience (years)']
        job_location = company_row['location']

        matched_users = match_users_to_company(required_skills, min_experience, job_location)

        if not matched_users.empty:
            st.success(f"🎯 Top candidates for {selected_company}")
            st.dataframe(matched_users[['skills', 'experience (years)', 'location', 'Score']])
        else:
            st.warning("⚠️ No matching candidates found. Try adjusting the criteria.")'''

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load Datasets
job_df = pd.read_csv("AI_Skill_Profiling_Dataset_100k_with_Future_Learning_Resources.csv")
user_df = pd.read_csv("cse_skills_dataset_200k.csv")
company_df = pd.read_csv("company_dataset.csv")

# Standardize column names
user_df.columns = user_df.columns.str.strip().str.lower()
company_df.columns = company_df.columns.str.strip().str.lower()

# Ensure lowercase for skills and locations
user_df['skills'] = user_df['skills'].astype(str).str.lower().str.strip()
user_df['location'] = user_df['location'].astype(str).str.lower().str.strip()

# Convert experience column to numeric
user_df['experience (years)'] = pd.to_numeric(user_df['experience (years)'], errors='coerce').fillna(0)

# TF-IDF Vectorizer for Job Prediction
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(job_df["Skills"]).toarray()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(job_df["Suggested Job"])
job_model = RandomForestClassifier(n_estimators=100, random_state=42)
job_model.fit(X, y)

# Initialize GPT-2 for Resume Generation
@st.cache_resource
def load_gpt_model():
    try:
        return pipeline('text-generation', model='gpt2')
    except Exception as e:
        st.error(f"Error loading GPT-2: {e}")
        return None

generator = load_gpt_model()

# TF-IDF Vectorizer for Talent Matching
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(", "), lowercase=True)
user_skill_matrix = vectorizer.fit_transform(user_df['skills'])

def predict_job(skills):
    """Predicts a job based on skills using trained model"""
    skills_tfidf = tfidf.transform([" ".join(skills)])
    if skills_tfidf.nnz == 0:
        return "No matching job found."
    
    prediction = job_model.predict(skills_tfidf)
    return label_encoder.inverse_transform(prediction)[0]

def generate_resume(form_data):
    """Generates a career objective using GPT-2"""
    if not generator:
        return "Career objective generation is unavailable."

    skills_text = ', '.join(form_data.get("skills", "").split(","))
    education = form_data.get("education", "a strong educational background")
    job_description = form_data.get("jobDescription", "varied professional experience")
    projects = form_data.get("projects", "several impactful projects")
    certifications = form_data.get("certifications", "relevant certifications")

    prompt = (
        f"I am passionate about my work and have a deep understanding of {skills_text}. "
        f"My education in {education} and experience in {job_description} have shaped my journey. "
        f"I have successfully completed projects such as {projects} and hold certifications in {certifications}. "
        f"I am excited about contributing my expertise."
    )

    try:
        generated = generator(prompt, max_length=100, num_return_sequences=1)
        return generated[0]['generated_text'].strip() if generated else "Unable to generate a career objective."
    except Exception as e:
        return f"Error: {e}"

def match_users_to_company(company_skills, company_experience, company_location, top_n=5):
    """Finds the best candidates for a given job"""
    company_skills_vector = vectorizer.transform([company_skills.lower()])
    similarity_scores = cosine_similarity(company_skills_vector, user_skill_matrix).flatten()

    # Apply experience and location weightage
    experience_weight = 0.3  
    experience_match = (user_df['experience (years)'] >= company_experience).astype(float) * experience_weight
    location_weight = 0.2  
    location_match = (user_df['location'] == company_location.lower()).astype(float) * location_weight

    final_scores = similarity_scores + experience_match + location_match
    user_indices = final_scores.argsort()[::-1][:top_n]

    recommended_users = user_df.iloc[user_indices].copy()
    recommended_users["Score"] = final_scores[user_indices]

    return recommended_users

# Streamlit UI
st.title("🔍 AI-Powered Job & Talent Matcher")

# Sidebar Navigation
menu = st.sidebar.radio("Choose an option:", ["Job Prediction", "Resume Generator", "Talent Matcher"])

if menu == "Job Prediction":
    st.header("💼 Predict Your Ideal Job")
    user_skills = st.text_area("Enter your skills (comma-separated):").strip()

    if st.button("Predict Job"):
        if user_skills:
            skills_list = [skill.strip() for skill in user_skills.split(",") if skill.strip()]
            predicted_job = predict_job(skills_list)
            st.success(f"🎯 Suggested Job: **{predicted_job}**")
        else:
            st.warning("⚠️ Please enter your skills.")

elif menu == "Resume Generator":
    st.header("📄 Generate Your Resume")
    
    name = st.text_input("Name:")
    phone = st.text_input("Phone Number:")
    email = st.text_input("Email:")
    skills = st.text_area("Skills (comma-separated):")
    education = st.text_area("Education:")
    job_desc = st.text_area("Job Experience:")
    projects = st.text_area("Projects:")
    certifications = st.text_area("Certifications:")

    if st.button("Generate Resume"):
        form_data = {
            "name": name, "phoneNumber": phone, "email": email,
            "skills": skills, "education": education,
            "jobDescription": job_desc, "projects": projects, "certifications": certifications
        }
        career_obj = generate_resume(form_data)

        st.subheader("📜 Generated Resume")
        st.text(f"**Name:** {name}")
        st.text(f"**Contact:** {phone} | **Email:** {email}")
        st.markdown(f"**Career Objective:** {career_obj}")
        st.text(f"**Skills:** {skills}")
        st.text(f"**Education:** {education}")
        st.text(f"**Projects:** {projects}")
        st.text(f"**Certifications:** {certifications}")

elif menu == "Talent Matcher":
    st.header("🔍 Find the Best Candidates for Your Job")

    # Company selection
    company_options = company_df['company name'].tolist()
    selected_company = st.selectbox("🏢 Select a company", company_options)

    if st.button("Find Candidates"):
        company_row = company_df[company_df['company name'] == selected_company].iloc[0]
        required_skills = company_row['required skills']
        min_experience = company_row['min experience (years)']
        job_location = company_row['location']

        matched_users = match_users_to_company(required_skills, min_experience, job_location)

        if not matched_users.empty:
            st.success(f"🎯 Top candidates for {selected_company}")
            st.dataframe(matched_users[['skills', 'experience (years)', 'location', 'Score']])
        else:
            st.warning("⚠️ No matching candidates found. Try adjusting the criteria.")

