import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import streamlit as st
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
import PyPDF2
import docx2txt
import re
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
from datetime import datetime
import sqlite3
import json

# Initialize the embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Database setup
def init_db():
    conn = sqlite3.connect('candidates.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS candidates
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  email TEXT,
                  phone TEXT,
                  skills TEXT,
                  experience TEXT,
                  education TEXT,
                  match_score REAL,
                  job_title TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Text extraction functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_text(file):
    if file.name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file.name.endswith('.docx'):
        return extract_text_from_docx(file)
    else:
        return file.read().decode('utf-8')

# NLP extraction functions
def extract_email(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not found"

def extract_phone(text):
    phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else "Not found"

def extract_name(text):
    lines = text.split('\n')
    # Usually name is in the first few lines
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 3 and len(line) < 50 and not '@' in line:
            # Simple heuristic: look for capitalized words
            words = line.split()
            if len(words) >= 2 and all(w[0].isupper() for w in words[:2] if w):
                return line
    return "Name not found"

def extract_skills(text):
    # Common tech skills list (expandable)
    skills_db = [
        'python', 'java', 'javascript', 'c++', 'sql', 'nosql', 'mongodb', 
        'postgresql', 'mysql', 'react', 'angular', 'vue', 'node.js', 'django',
        'flask', 'fastapi', 'machine learning', 'deep learning', 'nlp',
        'computer vision', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas',
        'numpy', 'data analysis', 'data science', 'aws', 'azure', 'gcp',
        'docker', 'kubernetes', 'git', 'agile', 'scrum', 'rest api', 'html',
        'css', 'typescript', 'scala', 'r', 'tableau', 'power bi', 'excel',
        'communication', 'leadership', 'project management', 'problem solving'
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill.title())
    
    return list(set(found_skills))

def extract_experience(text):
    # Look for years of experience
    exp_patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+)\+?\s*years?',
    ]
    
    for pattern in exp_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}+ years"
    
    # Look for date ranges to estimate experience
    date_pattern = r'(19|20)\d{2}\s*[-–]\s*(?:(19|20)\d{2}|present|current)'
    dates = re.findall(date_pattern, text, re.IGNORECASE)
    
    if dates:
        return f"~{len(dates)} positions found"
    
    return "Experience not specified"

def extract_education(text):
    education_keywords = [
        'bachelor', 'master', 'phd', 'doctorate', 'mba', 'b.tech', 'm.tech',
        'b.s', 'm.s', 'b.e', 'm.e', 'computer science', 'engineering',
        'university', 'college', 'institute'
    ]
    
    text_lower = text.lower()
    education = []
    
    for keyword in education_keywords:
        if keyword in text_lower:
            # Find the line containing the keyword
            for line in text.split('\n'):
                if keyword in line.lower():
                    education.append(line.strip())
                    break
    
    return ' | '.join(education[:3]) if education else "Education not specified"

def parse_resume(file):
    text = extract_text(file)
    
    return {
        'name': extract_name(text),
        'email': extract_email(text),
        'phone': extract_phone(text),
        'skills': extract_skills(text),
        'experience': extract_experience(text),
        'education': extract_education(text),
        'full_text': text
    }

def parse_job_description(job_text):
    return {
        'skills': extract_skills(job_text),
        'full_text': job_text
    }

def calculate_similarity(resume_data, job_data):
    # Create embeddings
    resume_embedding = model.encode(resume_data['full_text'])
    job_embedding = model.encode(job_data['full_text'])
    
    # Calculate cosine similarity
    similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
    
    # Skills overlap bonus
    resume_skills = set([s.lower() for s in resume_data['skills']])
    job_skills = set([s.lower() for s in job_data['skills']])
    
    if job_skills:
        skill_match = len(resume_skills.intersection(job_skills)) / len(job_skills)
    else:
        skill_match = 0
    
    # Weighted score: 70% semantic similarity + 30% skill match
    final_score = (similarity * 0.7 + skill_match * 0.3) * 100
    
    return final_score, list(resume_skills.intersection(job_skills))

def save_to_db(candidate, score, job_title):
    conn = sqlite3.connect('candidates.db')
    c = conn.cursor()
    c.execute('''INSERT INTO candidates 
                 (name, email, phone, skills, experience, education, match_score, job_title, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (candidate['name'], candidate['email'], candidate['phone'],
               json.dumps(candidate['skills']), candidate['experience'],
               candidate['education'], score, job_title,
               datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()



st.title("🤖 AI-Powered Resume Screening System")
st.markdown("### Automate your recruitment process with intelligent candidate matching")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    job_title = st.text_input("Job Title", "Software Engineer")
    min_score = st.slider("Minimum Match Score (%)", 0, 100, 50)
    
    st.markdown("---")
    st.header("📊 Database")
    if st.button("View All Candidates"):
        st.session_state.show_db = True
    if st.button("Clear Database"):
        conn = sqlite3.connect('candidates.db')
        c = conn.cursor()
        c.execute('DELETE FROM candidates')
        conn.commit()
        conn.close()
        st.success("Database cleared!")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Screen", "📈 Results", "💾 Database"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=300,
            placeholder="Enter the job requirements, required skills, experience, etc."
        )
    
    with col2:
        st.subheader("📄 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload candidate resumes (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        st.info(f"📁 {len(uploaded_files)} resume(s) uploaded")
    
    if st.button("🚀 Start Screening", type="primary", use_container_width=True):
        if not job_description:
            st.error("Please enter a job description!")
        elif not uploaded_files:
            st.error("Please upload at least one resume!")
        else:
            with st.spinner("🔍 Analyzing resumes..."):
                # Parse job description
                job_data = parse_job_description(job_description)
                
                # Parse all resumes and calculate scores
                results = []
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    resume_data = parse_resume(file)
                    score, matched_skills = calculate_similarity(resume_data, job_data)
                    
                    result = {
                        'name': resume_data['name'],
                        'email': resume_data['email'],
                        'phone': resume_data['phone'],
                        'skills': resume_data['skills'],
                        'matched_skills': matched_skills,
                        'experience': resume_data['experience'],
                        'education': resume_data['education'],
                        'score': round(score, 2),
                        'filename': file.name
                    }
                    
                    results.append(result)
                    
                    # Save to database
                    save_to_db(resume_data, round(score, 2), job_title)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Sort by score
                results.sort(key=lambda x: x['score'], reverse=True)
                
                # Store in session state
                st.session_state.results = results
                st.session_state.job_data = job_data
                
                st.success(f"✅ Screening complete! {len(results)} candidates analyzed.")

with tab2:
    if 'results' in st.session_state:
        results = st.session_state.results
        job_data = st.session_state.job_data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", len(results))
        with col2:
            qualified = len([r for r in results if r['score'] >= min_score])
            st.metric("Qualified Candidates", qualified)
        with col3:
            avg_score = np.mean([r['score'] for r in results])
            st.metric("Average Score", f"{avg_score:.1f}%")
        with col4:
            top_score = results[0]['score'] if results else 0
            st.metric("Top Score", f"{top_score:.1f}%")
        
        # Visualization
        st.subheader("📊 Score Distribution")
        df_viz = pd.DataFrame(results)
        fig = px.bar(df_viz, x='name', y='score', 
                     title='Candidate Match Scores',
                     color='score',
                     color_continuous_scale='RdYlGn',
                     labels={'score': 'Match Score (%)', 'name': 'Candidate'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top candidates
        st.subheader("🏆 Top Candidates")
        
        for idx, candidate in enumerate(results[:5], 1):
            with st.expander(f"#{idx} - {candidate['name']} ({candidate['score']}% match)", expanded=(idx==1)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📧 Email:** {candidate['email']}")
                    st.markdown(f"**📱 Phone:** {candidate['phone']}")
                    st.markdown(f"**💼 Experience:** {candidate['experience']}")
                    st.markdown(f"**🎓 Education:** {candidate['education']}")
                    
                    if candidate['matched_skills']:
                        st.markdown(f"**✅ Matched Skills ({len(candidate['matched_skills'])}):**")
                        skills_text = ", ".join([s.title() for s in candidate['matched_skills']])
                        st.success(skills_text)
                    
                    if candidate['skills']:
                        st.markdown(f"**🔧 All Skills:**")
                        st.info(", ".join(candidate['skills']))
                
                with col2:
                    # Score gauge
                    score_color = "🟢" if candidate['score'] >= 70 else "🟡" if candidate['score'] >= 50 else "🔴"
                    st.markdown(f"### {score_color} {candidate['score']}%")
                    st.progress(candidate['score'] / 100)
                    
                    st.markdown(f"**📄 File:** {candidate['filename']}")
        
        # Export functionality
        st.subheader("📥 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            df_export = pd.DataFrame(results)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON export
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="Download as JSON",
                data=json_str,
                file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.info("👆 Upload resumes and run screening in the 'Upload & Screen' tab to see results here.")

with tab3:
    st.subheader("💾 Candidate Database")
    
    conn = sqlite3.connect('candidates.db')
    df_db = pd.read_sql_query("SELECT * FROM candidates ORDER BY match_score DESC", conn)
    conn.close()
    
    if len(df_db) > 0:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            job_filter = st.selectbox("Filter by Job Title", ["All"] + list(df_db['job_title'].unique()))
        with col2:
            score_filter = st.slider("Minimum Score", 0, 100, 0)
        with col3:
            st.metric("Total Records", len(df_db))
        
        # Apply filters
        if job_filter != "All":
            df_db = df_db[df_db['job_title'] == job_filter]
        df_db = df_db[df_db['match_score'] >= score_filter]
        
        # Display table
        st.dataframe(
            df_db[['name', 'email', 'phone', 'job_title', 'match_score', 'timestamp']],
            use_container_width=True,
            hide_index=True
        )
        
        # Export database
        csv_db = df_db.to_csv(index=False)
        st.download_button(
            label="📥 Export Database as CSV",
            data=csv_db,
            file_name=f"candidate_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No candidates in database yet. Screen some resumes to populate the database!")

# Footer
st.markdown("---")
st.markdown("**🤖 AI Resume Screener** | Built with Streamlit, SentenceTransformers & scikit-learn")