from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.config import Config

# Initialize SQLAlchemy
db = SQLAlchemy()


def create_app(config_class=Config):
    # Initialize Flask
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)

    return app