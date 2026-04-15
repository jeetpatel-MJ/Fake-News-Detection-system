from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, Length
from flask_wtf import FlaskForm
from models import db, User, Submission, Report
from config import Config
import joblib
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os

# Load .env for API key
load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY')) if os.getenv('NEWS_API_KEY') else None

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Simple source credibility dictionary
source_credibility = {
    'reuters': 1.0, 'nytimes': 1.0, 'bbc': 1.0,
    'occupydemocrats': 0.0, 'palmerreport': 0.0,
}

def scrape_article(url):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'article'])
        text = ' '.join([elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True)])
        text = ' '.join(text.split())[:5000]
        if not text:
            raise ValueError("No readable text found.")
        return text
    except Exception as e:
        raise ValueError(f"Scraping failed: {str(e)}")

# Forms
class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class SubmissionForm(FlaskForm):
    url = StringField('URL (optional)')
    text = TextAreaField('Text (optional)')
    source = StringField('Source (optional)')
    submit = SubmitField('Analyze')

class ReportForm(FlaskForm):
    reason = TextAreaField('Reason for Report', validators=[DataRequired()])
    submit = SubmitField('Report')

# Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email already registered!', 'error')
            return redirect(url_for('register'))
        user = User(email=form.email.data, password=generate_password_hash(form.password.data), role='user')
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password!', 'error')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def dashboard():
    if not current_user.is_authenticated or current_user.id is None:
        flash('Authentication error. Please log in again.', 'error')
        return redirect(url_for('login'))
    
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    
    form = SubmissionForm()
    submission = None
    live_articles = None  # Now triggered only on submission
    if form.validate_on_submit():
        text = form.text.data.strip() or None
        url = form.url.data.strip() or None
        source = form.source.data.lower().strip() or None
        
        if url and not text:
            try:
                text = scrape_article(url)
                if not source:
                    source = url.split('/')[2].split('.')[0] if '/' in url else 'unknown'
            except ValueError as e:
                flash(str(e), 'error')
                return render_template('dashboard.html', form=form)
        
        if text:
            # Core Classification (unchanged)
            text_tfidf = vectorizer.transform([text])
            prob_real = model.predict_proba(text_tfidf)[0][1]
            prediction = 'Real' if prob_real > 0.5 else 'Fake'
            
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
            sentiment_label = 'Positive' if sentiment_polarity > 0 else 'Negative' if sentiment_polarity < 0 else 'Neutral'
            
            source_score = source_credibility.get(source, 0.5)
            overall_score = (prob_real + source_score) / 2
            if sentiment_polarity < 0:
                overall_score *= 0.8
            overall_verdict = 'Likely Real' if overall_score > 0.5 else 'Likely Fake'
            
            # Save submission
            submission_obj = Submission(
                user_id=current_user.id,
                text=text, 
                url=url, 
                prediction=prediction, 
                prob_real=prob_real
            )
            submission_obj.author = current_user
            db.session.add(submission_obj)
            db.session.flush()
            db.session.commit()
            
            submission = {
                'id': submission_obj.id,
                'prediction': prediction, 'prob_real': f'{prob_real:.2%}',
                'sentiment_label': sentiment_label, 'sentiment_polarity': f'{sentiment_polarity:.2f}',
                'sentiment_subjectivity': f'{sentiment_subjectivity:.2f}', 'source_score': f'{source_score:.2f}',
                'overall_verdict': overall_verdict, 'text_preview': text[:100] + '...'
            }
            flash('Analysis complete!', 'success')
            
            # New: Fetch Related Live News Based on Submission
            if newsapi:  # Check if API key is set
                # Use title-like query from text (first sentence or keywords)
                query = text.split('.')[0][:100]  # First sentence as query
                try:
                    articles = newsapi.get_everything(
                        q=query,
                        language='en',
                        sort_by='publishedAt',
                        page_size=3  # Small set for context
                    )
                    live_articles = []
                    for article in articles['articles']:
                        title_text = article['title'] + ' ' + (article['description'] or '')
                        text_tfidf = vectorizer.transform([title_text])
                        prob_real_live = model.predict_proba(text_tfidf)[0][1]
                        pred_live = 'Real' if prob_real_live > 0.5 else 'Fake'
                        s_live = article['source']['name'].lower()
                        score_live = source_credibility.get(s_live, 0.5)
                        blob_live = TextBlob(title_text)
                        sent_live = 'Positive' if blob_live.sentiment.polarity > 0 else 'Negative' if blob_live.sentiment.polarity < 0 else 'Neutral'
                        verdict_live = 'Likely Real' if (prob_real_live + score_live) / 2 > 0.5 else 'Likely Fake'
                        
                        live_articles.append({
                            'title': article['title'],
                            'url': article['url'],
                            'source': s_live,
                            'prediction': pred_live,
                            'prob_real': f'{prob_real_live:.2%}',
                            'sentiment_label': sent_live,
                            'overall_verdict': verdict_live
                        })
                    flash('Related live news fetched for context!', 'info')
                except Exception as e:
                    flash(f'Live news fetch failed: {str(e)}', 'warning')
            else:
                flash('Live news disabled (add NEWS_API_KEY to .env)', 'warning')
    
    history = Submission.query.filter_by(user_id=current_user.id).order_by(Submission.created_at.desc()).limit(10).all()
    
    return render_template('dashboard.html', form=form, submission=submission, history=history, live_articles=live_articles)
@app.route('/report/<int:sub_id>', methods=['POST'])
@login_required
def report(sub_id):
    sub = Submission.query.get(sub_id)
    if not sub:
        flash('Submission not found!', 'error')
        return redirect(url_for('dashboard'))
    
    form = ReportForm()
    if form.validate_on_submit():
        if sub.user_id != current_user.id:
            flash('Cannot report this submission!', 'error')
            return redirect(url_for('dashboard'))
        report_obj = Report(submission_id=sub_id, user_id=current_user.id, reason=form.reason.data)
        db.session.add(report_obj)
        db.session.commit()
        flash('Report submitted!', 'success')
    else:
        flash('Report form invalid—try again.', 'error')
    return redirect(url_for('dashboard'))

@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied!', 'error')
        return redirect(url_for('dashboard'))
    
    all_subs = Submission.query.order_by(Submission.created_at.desc()).limit(50).all()
    reports = Report.query.join(Submission).order_by(Report.created_at.desc()).all()
    users = User.query.all()
    
    total_subs = Submission.query.count()
    fake_count = Submission.query.filter_by(prediction='Fake').count()
    fake_rate = (fake_count / total_subs * 100) if total_subs > 0 else 0
    active_users = User.query.filter(User.submissions.any()).count()
    
    return render_template('admin.html', all_subs=all_subs, reports=reports, users=users,
                           total_subs=total_subs, fake_rate=fake_rate, active_users=active_users)

@app.route('/admin/review/<int:sub_id>', methods=['POST'])
@login_required
def admin_review(sub_id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    sub = Submission.query.get_or_404(sub_id)
    status = request.form.get('status')
    if status in ['verified_real', 'verified_fake']:
        sub.verified_status = status
        report = Report.query.filter_by(submission_id=sub_id).first()
        if report:
            report.status = 'reviewed'
            report.reviewed_by = current_user.id
        db.session.commit()
        flash('Review updated!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    user = User.query.get_or_404(user_id)
    if user.role == 'admin':
        flash('Cannot delete admin!', 'error')
        return redirect(url_for('admin_dashboard'))
    
    # Manually delete dependents
    reports_to_delete = Report.query.filter_by(user_id=user_id).all()
    for report in reports_to_delete:
        db.session.delete(report)
    
    submissions_to_delete = Submission.query.filter_by(user_id=user_id).all()
    for sub in submissions_to_delete:
        sub_reports = Report.query.filter_by(submission_id=sub.id).all()
        for sreport in sub_reports:
            db.session.delete(sreport)
        db.session.delete(sub)
    
    db.session.delete(user)
    db.session.commit()
    flash('User and all related submissions/reports deleted!', 'success')
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)