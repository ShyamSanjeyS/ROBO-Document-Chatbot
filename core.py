# core.py
import string
import random
import nltk
import spacy
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS

nltk.download('punkt')
nltk.download('wordnet')

class ChatbotContext:
    def __init__(self):
        self.text = ""
        self.sent_tokens = []
        self.cleaned_sentences = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self.chat_history = []

    def load_text(self, text):
        self.text = text
        self.sent_tokens = sent_tokenize(text)
        self.cleaned_sentences = self.preprocess_sentences(self.sent_tokens)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_sentences)

    @staticmethod
    def preprocess_sentences(sentences):
        translation = str.maketrans('', '', string.punctuation)
        return [s.lower().translate(translation) for s in sentences]

def greeting(sentence):
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def respond(context, user_input):
    query = context.vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(query, context.tfidf_matrix)
    max_sim_index = np.argmax(cosine_sim)
    if cosine_sim[0][max_sim_index] == 0:
        return "I beg your pardon? I'm not quite sure I got your meaning."
    return context.sent_tokens[max_sim_index]

def summarize_text(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text not in string.punctuation:
            word_frequencies[word.text.lower()] = word_frequencies.get(word.text.lower(), 0) + 1
    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_freq
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text.lower()]
    select_length = max(1, int(len(list(doc.sents)) * 0.3))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    return ' '.join([s.text for s in summary])
