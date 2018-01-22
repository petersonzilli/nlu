"""
author: peterson.zilli@gmail.com
"""
from models.shared import db

from models.intent import Intent

class Utterance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.Text, nullable=False)

    intent_id = db.Column(db.Integer, db.ForeignKey('intent.id'), nullable=False)

    def __repr__(self):
        return '<Utterance %r - %r>' % (self.intent.name, self.body)
