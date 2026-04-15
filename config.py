# config.py (New file: App configuration)
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hemen'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///fake_news.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False