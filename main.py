import openai
import PyPDF2
import docx
from textblob import TextBlob
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import logging
import json
import re
from werkzeug.utils import secure_filename
import os
import tempfile  # Import tempfile module

# Initialize Flask
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    UPLOAD_FOLDER = tempfile.mkdtemp()  # Use tempfile to create a temporary folder for uploads
    ALLOWED_EXTENSIONS = {'pdf', 'docx'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
app.config.from_object(Config)

# Ensure upload directory exists (although tempfile automatically creates the folder)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# OpenAI configuration
openai.api_type = "azure"
openai.api_base = "https://jobspringai.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "22Mj7xKp5fPvOKQDZ54xncvwHCUUt27nPBhmgI89k60HJ3do1kgTJQQJ99ALACYeBjFXJ3w3AAABACOGOy3V" # Using environment variable for security
DEPLOYMENT_NAME = "gpt-35-turbo-16k"

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_file):
        try:
            text = ''
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise

    @staticmethod
    def extract_text_from_docx(docx_file):
        try:
            doc = docx.Document(docx_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            raise

class TextAnalyzer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
    
    def extract_skills(self, text):
        doc = nlp(text)
        skills = []
        # Custom skill patterns (expand this list based on your needs)
        skill_patterns = [
            "python", "java", "javascript", "react", "angular", "vue",
            "docker", "kubernetes", "aws", "azure", "gcp", "sql", "nosql",
            "machine learning", "ai", "data science", "agile", "scrum"
        ]
        
        for token in doc:
            if token.text.lower() in skill_patterns:
                skills.append(token.text)
        return list(set(skills))

    def calculate_similarity(self, text1, text2):
        try:
            tfidf_matrix = self.tfidf.fit_transform([text1, text2])
            similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0][1]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def extract_experience(self, text):
        doc = nlp(text)
        experience = []
        
        # Pattern for finding years of experience
        experience_pattern = re.compile(r'\b(\d+)[\s-]*(year|yr)s?\b', re.IGNORECASE)
        matches = experience_pattern.finditer(text)
        
        total_years = 0
        for match in matches:
            years = int(match.group(1))
            total_years = max(total_years, years)
            
        return total_years

class AIAnalyzer:
    @staticmethod
    def get_completion(prompt, max_tokens=1000):
        try:
            response = openai.ChatCompletion.create(
                engine=DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return "Error generating analysis. Please try again."

    def analyze_resume(self, resume_text, job_desc_text):
        prompt = f"""
        Analyze this resume against the job description. Provide:
        1. Skills match analysis
        2. Experience relevance
        3. Missing critical requirements
        4. Suggested improvements
        5. Overall match percentage

        Resume: {resume_text[:2000]}...
        Job Description: {job_desc_text[:1000]}...
        """
        return self.get_completion(prompt)

    def generate_improvement_plan(self, analysis):
        prompt = f"""
        Based on this analysis, create a detailed improvement plan:
        {analysis}
        
        Include:
        1. Specific actions to take
        2. Skills to acquire
        3. Timeline for improvements
        4. Resources for skill development
        """
        return self.get_completion(prompt)

class ResumeAnalyzerAPI:
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.ai_analyzer = AIAnalyzer()
        self.doc_processor = DocumentProcessor()

    def analyze_document(self, resume_file, job_desc_text):
        try:
            # Extract text based on file type
            if resume_file.filename.endswith('.pdf'):
                resume_text = self.doc_processor.extract_text_from_pdf(resume_file)
            elif resume_file.filename.endswith('.docx'):
                resume_text = self.doc_processor.extract_text_from_docx(resume_file)
            else:
                raise ValueError("Unsupported file format")

            # Perform analysis
            skills = self.text_analyzer.extract_skills(resume_text)
            required_skills = self.text_analyzer.extract_skills(job_desc_text)
            similarity_score = self.text_analyzer.calculate_similarity(resume_text, job_desc_text)
            experience_years = self.text_analyzer.extract_experience(resume_text)
            
            # AI Analysis
            detailed_analysis = self.ai_analyzer.analyze_resume(resume_text, job_desc_text)
            improvement_plan = self.ai_analyzer.generate_improvement_plan(detailed_analysis)

            # Save the extracted text and results in temporary files
            with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
                temp_file.write(resume_text)
                resume_file_path = temp_file.name  # Store the path for later use

            result = {
                "skills_found": skills,
                "required_skills": required_skills,
                "missing_skills": list(set(required_skills) - set(skills)),
                "similarity_score": similarity_score,
                "experience_years": experience_years,
                "detailed_analysis": detailed_analysis,
                "improvement_plan": improvement_plan,
                "timestamp": datetime.now().isoformat(),
                "resume_text_path": resume_file_path  # Include the temp file path in the response
            }

            # Save analysis result in a temporary JSON file
            with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_json_file:
                json.dump(result, temp_json_file)
                result_file_path = temp_json_file.name  # Store the path for later use

            result["analysis_result_path"] = result_file_path  # Include the analysis file path in the response
            return result

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'resume' not in request.files:
            return jsonify({"error": "No resume file provided"}), 400
        
        resume_file = request.files['resume']
        if resume_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        job_desc_text = request.form.get('job_desc')
        if not job_desc_text:
            return jsonify({"error": "Job description is required"}), 400

        analyzer = ResumeAnalyzerAPI()
        result = analyzer.analyze_document(resume_file, job_desc_text)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Route error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File is too large"}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
