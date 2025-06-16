from datetime import datetime
from app import db


class AudioInput(db.Model):
    __tablename__ = 'audio_input_table'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    audio_file = db.Column(db.LargeBinary)
    filename = db.Column(db.String(255))
    processed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    results = db.relationship('ThreatAnalysisResult', backref='audio_input', lazy=True)


class ThreatAnalysisResult(db.Model):
    __tablename__ = 'threat_analysis_results'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    audio_id = db.Column(db.Integer, db.ForeignKey('audio_input_table.id'))
    transcript = db.Column(db.Text)
    threat_type = db.Column(db.String(50))
    confidence = db.Column(db.Numeric(3, 2))  # DECIMAL(3,2) in PostgreSQL
    severity = db.Column(db.String(20))
    analysis = db.Column(db.Text)
    keywords = db.Column(db.JSON)  # JSONB in PostgreSQL
    urgent = db.Column(db.Boolean, default=False)
    recommended_action = db.Column(db.Text)
    audio_file = db.Column(db.LargeBinary)
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)