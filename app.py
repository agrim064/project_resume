import os
import re
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# File upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Preprocessing function
def preprocess(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_description = request.form['job_description']
        job_description = preprocess(job_description)

        # Save uploaded files
        files = request.files.getlist('resumes')
        resume_texts = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Extract text from file
                if filename.endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                else:
                    with open(file_path, 'r') as f:
                        text = f.read()
                resume_texts.append(preprocess(text))

        # TF-IDF and cosine similarity
        all_texts = [job_description] + resume_texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Rank resumes
        ranked_resumes = sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)
        rankings = [{"rank": i+1, "filename": files[index].filename, "score": round(score, 4)}
                    for i, (index, score) in enumerate(ranked_resumes)]

        return render_template('results.html', rankings=rankings)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
