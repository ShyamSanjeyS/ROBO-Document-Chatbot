import os
import string
import random
import nltk
import spacy
import pyttsx3
import speech_recognition as sr
import numpy as np
import fitz  # PyMuPDF
from docx import Document
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest

# Initialize
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Speak function
def speak(text):
    print("ROBO:", text)
    engine.say(text)
    engine.runAndWait()

# Listen function
def listen():
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand. Please try again.")
        return ""
    except sr.RequestError:
        speak("Sorry, there was an error with the speech service.")
        return ""

# Chatbot core
class ChatbotContext:
    def __init__(self):
        self.text = ""
        self.sent_tokens = []
        self.cleaned_sentences = []
        self.tfidf_matrix = None
        self.vectorizer = None

    def load_text(self, text):
        self.text = text
        self.sent_tokens = sent_tokenize(text)
        self.cleaned_sentences = self._preprocess(self.sent_tokens)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_sentences)

    def _preprocess(self, sentences):
        table = str.maketrans('', '', string.punctuation)
        return [s.lower().translate(table) for s in sentences]

def extract_text(filepath):
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif filepath.endswith(".pdf"):
        with open(filepath, "rb") as f:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        speak("Unsupported file format.")
        return ""

def summarize(text):
    doc = nlp(text)
    word_freq = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text not in string.punctuation:
            word_freq[word.text.lower()] = word_freq.get(word.text.lower(), 0) + 1
    max_freq = max(word_freq.values(), default=1)
    word_freq = {k: v / max_freq for k, v in word_freq.items()}
    sent_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_freq:
                sent_scores[sent] = sent_scores.get(sent, 0) + word_freq[word.text.lower()]
    summary = nlargest(max(1, int(len(list(doc.sents)) * 0.3)), sent_scores, key=sent_scores.get)
    return ' '.join([s.text for s in summary])

def greeting(sentence):
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    GREETING_RESPONSES = ["hi", "hey", "hello there", "nice to meet you"]
    for word in sentence.lower().split():
        if word in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def respond(context, user_input):
    query = context.vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(query, context.tfidf_matrix)
    max_sim_index = np.argmax(cosine_sim)
    if cosine_sim[0][max_sim_index] == 0:
        return "I'm not sure I understood that."
    return context.sent_tokens[max_sim_index]

# Voice chatbot flow
def voice_chatbot():
    speak("Welcome to Robo, your document assistant.")
    path = input("Enter the path to the document (.txt, .pdf, .docx): ").strip()
    if not os.path.exists(path):
        speak("File not found.")
        return

    text = extract_text(path)
    if not text:
        return

    context = ChatbotContext()
    context.load_text(text)
    speak("Document loaded and ready. You can start asking questions. Say 'summarize' for summary or 'exit' to quit.")

    while True:
        query = listen()
        if not query:
            continue
        if "exit" in query.lower():
            speak("Goodbye!")
            break
        elif "summarize" in query.lower():
            summary = summarize(context.text)
            speak("Here's the summary.")
            speak(summary)
        else:
            greet = greeting(query)
            answer = greet if greet else respond(context, query)
            speak(answer)

if __name__ == "__main__":
    voice_chatbot()
