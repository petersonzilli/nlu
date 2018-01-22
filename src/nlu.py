# -*- coding: utf-8 -*-
"""
author: peterson.zilli@gmail.com

TODO:
    * Read and apply info from https://stackoverflow.com/questions/9692962/flask-sqlalchemy-import-context-issue/9695045#9695045
    * Read and apply info from http://piotr.banaszkiewicz.org/blog/2012/06/29/flask-sqlalchemy-init_app/
"""
import os.path

from flask import abort, flash, Flask, jsonify, url_for, redirect, \
                    render_template, request, send_from_directory, Markup
from flask_sqlalchemy import SQLAlchemy

from models.shared import db
from models.intent import Intent
from models.utterance import Utterance

app = Flask(__name__, static_url_path='')
app.config.from_pyfile('nlu.cfg')
db.init_app(app)

#next line solves an error on contexts http://flask-sqlalchemy.pocoo.org/2.3/contexts/
app.app_context().push() 

#****************************************************
# Make Dirs
#****************************************************
import errno    
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

import shutil
def rmdir(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        print(exc)
        if exc.errno == errno.ENOENT and not os.path.isdir(path):
            pass
        else:
            raise

#****************************************************
# Make Dirs
#****************************************************

if not os.path.isfile('nlu.db'):
    print('creating database file...')
    db.create_all()

# serve static files from directory
@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('static/js', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('static/css', path)

# routes
@app.route('/', methods=["GET"])
@app.route('/intents', methods=["GET"])
@app.route('/intents/', methods=["GET"])
def index():
    intents = Intent.query.all()
    return render_template('intents.html', intents=Intent.query.order_by(Intent.name.asc()).all())

@app.route('/intents/addIntentDialogBox', methods=['GET'])
def intent_add_dialog_box():
    return render_template('intents_add_dialog_box.html')

@app.route('/intents/detail/<int:intent_id>', methods=["GET"])
def utterances_home(intent_id=None):
    intent = Intent.query.filter(Intent.id == intent_id).first()
    if intent is None:
        return jsonify({'ok':False, 'message': 'Intent "'+ intent_name +'" not found in the database.'})
    return render_template('utterances.html', intent=intent)

@app.route('/intents/addutterance/', methods=["POST"])
def utterances_add():
    req_data = request.get_json()
    intent_name = req_data['intent_name']
    utterance_body = req_data['utterance_body']
    intent = Intent.query.filter(Intent.name == intent_name).first()
    if intent is None:
        return jsonify({'ok':False, 'message': 'Intent "'+ intent_name +'" not found in the database.'})
        # filter empty cases
    if len(utterance_body.replace(' ','').replace('\t','')) == 0:
        return jsonify({'ok':False, 'message': 'Utterance cannot be only spaces or tabs'})
    # Filter duplicated utterance 
    this_utterance_body_already_registered = Utterance.query.filter(Utterance.body == utterance_body)
    if this_utterance_body_already_registered.count() > 0:
        print("Utterance name already registered: %r" % this_utterance_body_already_registered.first().body)
        return jsonify({'ok': False, 'message': 'Utterance already registered: %r' % this_utterance_body_already_registered.first().body })
    new_utterance = Utterance(body=utterance_body, intent=intent)
    db.session.add(new_utterance)
    db.session.commit()
    print("added utterance <" + utterance_body + "> into intent <" + intent_name + ">")
    return jsonify({'ok':True})
    return render_template('utterances.html', intent=intent)

@app.route('/intents/deleteutterance/<int:utterance_id>', methods=["GET"])
def utterance_remove(utterance_id):
    utterance = Utterance.query.filter(Utterance.id == utterance_id).first()
    #print("deleting utterance: " + utterance.body)
    db.session.delete(utterance)
    db.session.commit()
    #print("deleted utterance: " + utterance.body)
    return jsonify({'ok':True})


@app.route('/intents/add/', methods=["GET"])
@app.route('/intents/add/<intent_name>', methods=["GET"])
def intent_add(intent_name=""):
    #print("adding intent: " + intent_name)
    # filter empty cases
    if len(intent_name.replace(' ','').replace('\t','')) == 0:
        return jsonify({'ok':False, 'message': 'Intent name cannot be only spaces or tabs'})
    # Filter duplicates
    this_intent_name_already_registered = Intent.query.filter(Intent.name == intent_name).count()
    if this_intent_name_already_registered > 0:
        print("intent name already registered.")
        return jsonify({'ok': False, 'message': 'Intent Name already registered.'})
    new_intent = Intent(name=intent_name)
    db.session.add(new_intent)
    db.session.commit()
    #print("added intent: " + intent_name)
    return jsonify({'ok':True})

@app.route('/intents/delete/<int:intent_id>', methods=["GET"])
def intent_remove(intent_id):
    intent = Intent.query.filter(Intent.id == intent_id).first()
    #print("deleting intent: " + intent.name)
    ## removing all utterances first:
    for utterance in intent.utterances:
        utterance = Utterance.query.filter(Utterance.id == utterance_id).first()
        #print("deleting utterance: " + utterance.body)
        db.session.delete(utterance)
        db.session.commit()
    ## removing intent
    db.session.delete(intent)
    db.session.commit()
    #print("deleted intent: " + intent.name)
    return jsonify({'ok':True})

@app.route('/extractscreen', methods=["GET"])
def data_extractscreen():
    extraction = {}
    intents = Intent.query.order_by(Intent.name.asc()).all()
    utterances = Utterance.query.order_by(Utterance.body.asc()).all()
    extraction['intents'] = [ {'name' : i.name} for i in intents ]
    extraction['utterances'] = [ {'text' : u.body, 'intent': u.intent.name } for u in utterances ]
    return jsonify(extraction)


@app.route('/extractfiles', methods=["GET"])
def data_extractfiles():
        #try:
        rmdir('./data')
        mkdir_p('./data')

        intents = Intent.query.order_by(Intent.name.asc()).all()
        for intent in intents:
            filename = './data/' + intent.name + '.txt'
            with open(filename, 'wb') as fp:
                for u in intent.utterances:
                    fp.write((u.body + '\n').encode("UTF-8"))

        return jsonify({'ok':True})
    #except Exception as e:
    #    return jsonify({'ok':False, 'message': str(e)})

import subprocess

def train_intents():
    """
    Return a string that is the output from subprocess
    """
    out = subprocess.check_output(["C:\\Users\\peter\\Anaconda3\\Scripts\\activate.bat", "NLU", "&&", "C:\\Users\\peter\\Anaconda3\\python.exe", "train.py"])
    return out.decode('latin_1').replace('\r\n', '<br>')

@app.route('/trainintents', methods=["GET"])
def trainintents():
    return render_template('train_intents.html', subprocess_output= Markup(train_intents()))

def test_intents(user_input=""):
    """
    Return a string that is the output from subprocess
    """
    #out = subprocess.check_output(["C:\\Users\\peter\\Anaconda3\\Scripts\\activate.bat", "NLU", "&&", "C:\\Users\\peter\\Anaconda3\\python.exe", "eval.py", "--eval_train=False", '--checkpoint_dir=".\\runs\\1516586601\\checkpoints\\"', '--x="'+user_input + '"'])
    out = subprocess.check_output(["C:\\Users\\peter\\Anaconda3\\python.exe", "eval.py", "--eval_train=False", '--x="'+user_input + '"'])
    print(out)
    return out.decode('latin_1', errors='replace').replace('\r\n', '<br>')


@app.route('/testintents/<user_input>', methods=["GET"])
def testintents(user_input):
    return render_template('train_intents.html', subprocess_output= Markup(test_intents(user_input)))


if __name__ == "__main__":
    app.run()
