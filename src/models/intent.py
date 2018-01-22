"""
author: peterson.zilli@gmail.com
"""
from models.shared import db

class Intent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    utterances = db.relationship('Utterance', backref='intent', lazy='dynamic')
    
    def __repr__(self):
        return '<Intent %r>' % self.name
