from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
import pickle
import re
import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
from chatterbot import ChatBot
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.
    Args:
    embeddings_path - path to the embeddings file.
    Returns:
    embeddings - dict mapping words to vectors;
    embeddings_dim - dimension of the vectors.
    """
    
    embeddings = dict()
    for line in open(embeddings_path, encoding='utf-8'):
        row = line.strip().split('\t')
        embeddings[row[0]] = np.array(row[1:], dtype=np.float32)
    embeddings_dim = embeddings[list(embeddings)[0]].shape[0]
    
    return embeddings, embeddings_dim


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim).reshape(1,-1)
        best_thread = pairwise_distances_argmin(question_vec,thread_embeddings)
        
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think this is about %s. This might be of help: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        
        #init chatbot
        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Create a new trainer for the chatbot
        self.chitchat_bot = ChatBot('Nim Obvious')
        trainer = ChatterBotCorpusTrainer(self.chitchat_bot)

        # Train based on the english corpus
        trainer.train("chatterbot.corpus.english")

    
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""
        
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)
        
        
        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chitchat_bot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict( features)[0]
            #print(tag)

            thread_id = self.thread_ranker.get_best_thread(question, tag)[0]
        
            return self.ANSWER_TEMPLATE % (tag, thread_id)

    
def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    result = np.zeros(dim)
    cnt = 0
    words = question.split()
    for word in words:
        if word in embeddings:
            result += np.array(embeddings[word])
            cnt += 1
    if cnt != 0:
        result /= cnt
    return result

dialogue_manager = DialogueManager(RESOURCE_PATH)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/new_post', methods=['POST','GET'])
def new_post():

    if request.method == 'POST':
        
        question = request.form['Question']
        answer = dialogue_manager.generate_answer(question)
        return render_template('result.html', answer = answer)
        

    else:
        return render_template('new_post.html')






if __name__ == "__main__":
    app.run(debug=True)