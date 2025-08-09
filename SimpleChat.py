import os
import random
import string
import warnings
import nltk
import spacy
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings("ignore")

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

# --- Load the document text ---
def load_text():
    file_path = input("Enter the text file path: ")
    if not os.path.exists(file_path):
        print("File not found!")
        exit()

    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        text = f.read()
    return text

text = load_text()
sent_tokens = sent_tokenize(text)

# --- Preprocess sentences ---
def preprocess_text(sentences):
    translation = str.maketrans('', '', string.punctuation)
    cleaned = [s.lower().translate(translation) for s in sentences]
    return cleaned

cleaned_sentences = preprocess_text(sent_tokens)

# --- TF-IDF Model ---
TfidfVec = TfidfVectorizer(stop_words='english')
tfidf_matrix = TfidfVec.fit_transform(cleaned_sentences)
X = pd.DataFrame(tfidf_matrix.toarray(), columns=TfidfVec.get_feature_names_out(), dtype='float32')

# --- Greeting Handler ---
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

# --- Generate Response ---
def respond(user_input):
    query = TfidfVec.transform([user_input])
    cosine_sim = cosine_similarity(query, tfidf_matrix)
    max_sim_index = np.argmax(cosine_sim)

    if cosine_sim[0][max_sim_index] == 0:
        return "BOT: I beg your pardon? I\'m not quite sure I got your meaning."
    return sent_tokens[max_sim_index]

# --- Summarize Text ---
def summarize_text(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text not in string.punctuation:
            word_frequencies[word.text.lower()] = word_frequencies.get(word.text.lower(), 0) + 1

    max_freq = max(word_frequencies.values())
    word_frequencies = {k: v / max_freq for k, v in word_frequencies.items()}

    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text.lower()]

    select_length = max(1, int(len(list(doc.sents)) * 0.3))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    return ' '.join([s.text for s in summary])

# --- Chatbot Loop ---
def chatbot():
    print("ROBO: My name is Robo. I will answer your queries about the document. If you want to exit, type 'bye'.")
    while True:
        user_input = input("YOU: ").strip().lower()
        if user_input == 'bye':
            print("ROBO: Bye! Take care.")
            break
        elif user_input in ('thanks', 'thank you'):
            print("ROBO: You are welcome.")
        elif user_input == "~summarize":
            print("ROBO: " + summarize_text(text))
        else:
            greet = greeting(user_input)
            if greet:
                print("ROBO: " + greet)
            else:
                print("ROBO:", respond(user_input))

if __name__ == "__main__":
    chatbot()
