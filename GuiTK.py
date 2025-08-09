import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
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
from docx import Document
import fitz  # PyMuPDF for PDF

# Setup
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
        self.cleaned_sentences = self._preprocess(self.sent_tokens)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_sentences)

    @staticmethod
    def _preprocess(sentences):
        trans = str.maketrans('', '', string.punctuation)
        return [s.lower().translate(trans) for s in sentences]

def greeting(sentence):
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "Nice to meet you!"]
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def respond(context, user_input):
    query = context.vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(query, context.tfidf_matrix)
    max_sim_index = np.argmax(cosine_sim)
    if cosine_sim[0][max_sim_index] == 0:
        return "I beg your pardon? I'm not quite sure I understood."
    return context.sent_tokens[max_sim_index]

def summarize(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    word_freq = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text not in string.punctuation:
            word_freq[word.text.lower()] = word_freq.get(word.text.lower(), 0) + 1
    max_freq = max(word_freq.values())
    word_freq = {k: v / max_freq for k, v in word_freq.items()}
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word.text.lower()]
    summary_len = max(1, int(len(list(doc.sents)) * 0.3))
    summary = nlargest(summary_len, sentence_scores, key=sentence_scores.get)
    return " ".join([s.text for s in summary])

def extract_text(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif path.endswith(".pdf"):
        with open(path, "rb") as f:
            pdf = fitz.open(stream=f.read(), filetype="pdf")
            return "\n".join([p.get_text() for p in pdf])
    elif path.endswith(".docx"):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

class ChatbotApp:
    def __init__(self, root):
        self.context = ChatbotContext()
        self.root = root
        self.root.title("Robo - Document Chatbot")
        self.root.geometry("780x600")
        self.root.configure(bg="#f0f2f5")

        # Title
        title = tk.Label(root, text="🤖 Robo - Document Chatbot", font=("Segoe UI", 18, "bold"), bg="#f0f2f5", fg="#333")
        title.pack(pady=10)

        # Chat Frame
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 11), height=20)
        self.chat_area.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.configure(state='disabled')

        # Input Frame
        input_frame = tk.Frame(root, bg="#f0f2f5")
        input_frame.pack(padx=20, pady=5, fill=tk.X)

        self.entry = tk.Entry(input_frame, font=("Segoe UI", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry.bind("<Return>", self.process_input)

        self.load_btn = tk.Button(input_frame, text="📂 Load Document", command=self.load_document, bg="#0078D7", fg="white", font=("Segoe UI", 10, "bold"))
        self.load_btn.pack(side=tk.LEFT)

        self.summarize_btn = tk.Button(input_frame, text="📄 Summarize", command=self.summarize_text, bg="#28a745", fg="white", font=("Segoe UI", 10, "bold"))
        self.summarize_btn.pack(side=tk.LEFT, padx=(10, 0))

        self._bot_print("Hi, I'm Robo 🤖. Load a document and ask me questions!")

    def _bot_print(self, msg):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, f"ROBO: {msg}\n\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.see(tk.END)

    def _user_print(self, msg):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, f"YOU: {msg}\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.see(tk.END)

    def process_input(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self._user_print(user_input)
        self.entry.delete(0, tk.END)

        if not self.context.text:
            self._bot_print("Please load a document first.")
            return

        greet = greeting(user_input)
        if greet:
            response = greet
        else:
            response = respond(self.context, user_input)

        self._bot_print(response)

    def summarize_text(self):
        if not self.context.text:
            self._bot_print("Please load a document first.")
            return
        summary = summarize(self.context.text)
        self._bot_print("📄 Summary:\n" + summary)

    def load_document(self):
        file_path = filedialog.askopenfilename(filetypes=[("Documents", "*.txt *.pdf *.docx")])
        if not file_path:
            return
        try:
            text = extract_text(file_path)
            if not text.strip():
                raise ValueError("Empty content")
            self.context.load_text(text)
            self._bot_print("✅ Document loaded successfully. Ask your question!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load document:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
