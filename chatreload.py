import os
import random
import string
import warnings
import nltk
import spacy
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings("ignore")

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

    def load_text_file(self, path):
        if not os.path.exists(path):
            print("File not found!")
            return False
        with open(path, 'r', encoding='utf8', errors='ignore') as f:
            self.text = f.read()
        self.sent_tokens = sent_tokenize(self.text)
        self.cleaned_sentences = self.preprocess_sentences(self.sent_tokens)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_sentences)
        print("Document loaded successfully.")
        return True

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
        return "BOT: I beg your pardon? I'm not quite sure I got your meaning."
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


def chatbot():
    context = ChatbotContext()

    while True:
        file_path = input("Enter the text file path (or type 'skip' to keep current): ").strip()
        if file_path.lower() == 'exit':
            print("ROBO: Exiting chatbot.")
            return
        if file_path.lower() != 'skip' and not context.load_text_file(file_path):
            continue
        else:
            break

    print("ROBO: I am ready! Type your question, '~summarize' to summarize, or 'reload' to change the document. Type 'bye' to exit.")

    while True:
        user_input = input("YOU: ").strip()
        context.chat_history.append(("YOU", user_input))

        if user_input.lower() == 'bye':
            print("ROBO: Bye! Take care.")
            break
        elif user_input.lower() in ('thanks', 'thank you'):
            print("ROBO: You are welcome.")
        elif user_input.lower() == '~summarize':
            print("ROBO: " + summarize_text(context.text))
        elif user_input.lower() == 'reload':
            chatbot()  # Restart chatbot for reload
            break
        else:
            greet = greeting(user_input)
            if greet:
                print("ROBO: " + greet)
            else:
                response = respond(context, user_input)
                context.chat_history.append(("ROBO", response))
                print("ROBO: " + response)

if __name__ == "__main__":
    chatbot()
