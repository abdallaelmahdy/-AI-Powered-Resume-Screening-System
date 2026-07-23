#  AI-Powered Resume Screening Application

An intelligent HR automation tool that uses Natural Language Processing (NLP) and machine learning to automatically screen resumes, extract candidate information, and rank applicants based on job requirements.

## ✨ Features

- **Multi-format Support**: Upload resumes in PDF, DOCX, or TXT format
- **Intelligent Extraction**: Automatically extracts name, email, phone, skills, experience, and education
- **Semantic Matching**: Uses SentenceTransformers for deep semantic similarity analysis
- **Skill Matching**: Identifies and matches candidate skills with job requirements
- **Smart Ranking**: Combines semantic similarity (70%) and skill matching (30%) for accurate scoring
- **Interactive Dashboard**: Beautiful Streamlit interface with real-time results
- **Data Persistence**: SQLite database to store and track all candidates
- **Visualizations**: Interactive charts showing score distributions
- **Export Options**: Download results as CSV or JSON
- **Batch Processing**: Screen multiple resumes simultaneously

## 🛠️ Technology Stack

- **Python 3.8+**
- **Streamlit**: Web interface
- **SentenceTransformers**: Semantic embeddings (all-MiniLM-L6-v2 model)
- **scikit-learn**: Cosine similarity calculations
- **PyPDF2**: PDF text extraction
- **docx2txt**: DOCX text extraction
- **Plotly**: Interactive visualizations
- **SQLite**: Database storage

## 📦 Installation

### Step 1: Clone or Download the Project

```bash
# Create project directory
mkdir resume-screener
cd resume-screener
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Resumes and Job Description

1. Navigate to the **"Upload & Screen"** tab
2. Paste the job description in the left text area
3. Upload one or multiple resume files (PDF, DOCX, or TXT) using the file uploader
4. Click **"Start Screening"** to begin the analysis

### 2. View Results

1. Switch to the **"Results"** tab after screening completes
2. View summary metrics: Total candidates, qualified candidates, average score
3. See the score distribution chart
4. Expand top candidates to view detailed information:
   - Contact information
   - Matched skills
   - All extracted skills
   - Experience and education
   - Match score percentage

### 3. Export Results

- Click **"Download as CSV"** to export results in spreadsheet format
- Click **"Download as JSON"** for structured data export

### 4. Database Management

1. Navigate to the **"Database"** tab
2. View all historical screening records
3. Filter by job title or minimum score
4. Export the entire database as CSV

## 🎯 How It Works

### Text Extraction
- Extracts text from uploaded resume files
- Handles multiple formats (PDF, DOCX, TXT)

### Information Extraction
Uses regex patterns and NLP techniques to extract:
- **Name**: From header section
- **Email**: Email pattern matching
- **Phone**: Phone number pattern matching
- **Skills**: Matches against comprehensive skills database
- **Experience**: Identifies years of experience
- **Education**: Extracts education details

### Matching Algorithm

The system uses a hybrid approach:

1. **Semantic Similarity (70% weight)**
   - Generates embeddings for resume and job description
   - Uses SentenceTransformers (all-MiniLM-L6-v2)
   - Calculates cosine similarity between embeddings

2. **Skill Matching (30% weight)**
   - Extracts required skills from job description
   - Identifies matching skills in candidate resume
   - Calculates overlap percentage

3. **Final Score**
   ```
   Final Score = (Semantic Similarity × 0.7 + Skill Match × 0.3) × 100
   ```

### Ranking
- Candidates are ranked by final score (highest to lowest)
- Top 5 candidates are highlighted in results
- Visual indicators show score quality (🟢 70%+, 🟡 50-70%, 🔴 <50%)

## 🔧 Customization

### Adding More Skills

Edit the `skills_db` list in the `extract_skills()` function:

```python
skills_db = [
    'python', 'java', 'javascript',
    # Add your custom skills here
    'your_skill_1', 'your_skill_2'
]
```

### Adjusting Scoring Weights

Modify the weights in the `calculate_similarity()` function:

```python
# Current: 70% semantic, 30% skills
final_score = (similarity * 0.7 + skill_match * 0.3) * 100

# Example: 50% semantic, 50% skills
final_score = (similarity * 0.5 + skill_match * 0.5) * 100
```

### Changing Minimum Score Threshold

Use the sidebar slider or modify the default:

```python
min_score = st.slider("Minimum Match Score (%)", 0, 100, 50)  # Default: 50
```

## 📊 Database Schema

The SQLite database stores candidates with the following fields:

- `id`: Primary key
- `name`: Candidate name
- `email`: Email address
- `phone`: Phone number
- `skills`: JSON array of skills
- `experience`: Experience summary
- `education`: Education details
- `match_score`: Calculated match percentage
- `job_title`: Associated job title
- `timestamp`: Screening date and time

## 🚀 Deployment Options

### Local Deployment
Already covered in installation steps above.

### Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t resume-screener .
docker run -p 8501:8501 resume-screener
```

## 🐛 Troubleshooting

### Model Download Issues
If SentenceTransformers fails to download the model:
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### PDF Extraction Errors
Some PDFs may have protection or unusual formatting. Try:
- Converting PDF to DOCX first
- Using online PDF converters
- Saving as text file

### Memory Issues
For large batches:
- Process resumes in smaller batches
- Use a lighter model
- Increase system memory allocation

## 📝 Sample Job Description

```
Job Title: Senior Software Engineer

We are looking for an experienced software engineer with:
- 5+ years of Python development experience
- Strong knowledge of machine learning and NLP
- Experience with Django or Flask frameworks
- Proficiency in SQL and NoSQL databases
- AWS or Azure cloud experience
- Excellent problem-solving and communication skills
- Bachelor's degree in Computer Science or related field

Responsibilities:
- Design and develop scalable applications
- Collaborate with cross-functional teams
- Mentor junior developers
- Implement best practices and code reviews
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Enhanced entity extraction using spaCy
- Support for more file formats
- Advanced filtering and search
- Integration with ATS systems
- Email notification system
- Multi-language support

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- SentenceTransformers by UKPLab
- Streamlit for the amazing framework
- The open-source NLP community

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check existing documentation
- Review the troubleshooting section



