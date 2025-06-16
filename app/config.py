import os
from datetime import timedelta


class Config:
    # Flask Configuration
    SECRET_KEY = 'your-secret-key-here'

    # Database Configuration - update these with your actual credentials
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:1234@localhost:5432/postgres'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File Upload Configuration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # API Keys
    SARVAM_AI_API = "23ad1cf2-7480-445f-9130-72cd748a106a"
    GEMINI_API_KEY = "AIzaSyCPdabdtJxvnzMAPrinr_uxJJaojhcnQeY"

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)