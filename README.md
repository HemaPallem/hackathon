#AI-Powered Job & Talent Matcher
This project leverages machine learning and natural language processing to help users predict ideal jobs, generate resumes, and match candidates to companies based on their skills and experience.

Features
Job Prediction: Predicts the ideal job based on the skills you input.
Resume Generator: Generates a career objective for a resume based on user inputs like skills, education, and job experience.
Talent Matcher: Matches job seekers to companies based on their skill set, experience, and location.
Technologies Used
Streamlit: For creating the web application interface.
Pandas: For data manipulation and analysis.
Scikit-learn: For machine learning models and vectorization (TF-IDF).
Transformers (Hugging Face): For text generation using GPT-2 for resume generation.
Cosine Similarity: For matching job seekers with companies.
Installation
Clone this repository:

bash
Copy
Edit
git clone https://github.com/your-username/ai-job-talent-matcher.git
cd ai-job-talent-matcher
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Open your browser and go to the provided local address to access the application.

Files
app.py: Main application file that runs the Streamlit interface.
AI_Skill_Profiling_Dataset_100k_with_Future_Learning_Resources.csv: Job dataset used for job prediction.
cse_skills_dataset_200k.csv: User dataset used for talent matching.
company_dataset.csv: Company dataset containing job requirements for talent matching.
requirements.txt: List of Python dependencies for the project.
README.md: Documentation for the project.
How to Use
Job Prediction
Select "Job Prediction" from the sidebar.
Enter a list of your skills (comma-separated).
Click "Predict Job" to see your suggested job based on your skills.
Resume Generator
Select "Resume Generator" from the sidebar.
Enter your details, including name, phone number, email, skills, education, job experience, projects, and certifications.
Click "Generate Resume" to get a career objective generated based on the provided data.
Talent Matcher
Select "Talent Matcher" from the sidebar.
Enter your company details, including company name, required skills, minimum experience, and job location.
Click "Find Candidates" to see the top matched candidates based on the provided criteria.
Contribution
Feel free to fork this repository and submit pull requests if you'd like to contribute to the project.

