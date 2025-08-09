import typer
import os
import string
import random
import pickle
import nltk
import spacy
import numpy as np
from pathlib import Path
from typing import Optional
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS
from docx import Document
import fitz  # PyMuPDF for PDF

app = typer.Typer()
nltk.download('punkt')
nltk.download('wordnet')

CONTEXT_FILE = "context.pkl"

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

    @staticmethod
    def _preprocess(sentences):
        trans = str.maketrans('', '', string.punctuation)
        return [s.lower().translate(trans) for s in sentences]

def save_context(context: ChatbotContext):
    with open(CONTEXT_FILE, 'wb') as f:
        pickle.dump(context, f)

def load_context() -> Optional[ChatbotContext]:
    if not os.path.exists(CONTEXT_FILE):
        return None
    with open(CONTEXT_FILE, 'rb') as f:
        return pickle.load(f)

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
    else:
        typer.echo("❌ Unsupported file type.")
        raise typer.Exit()

def greeting(sentence):
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "Nice to meet you!"]
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def respond(context: ChatbotContext, user_input: str):
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

@app.command()
def load(file: Path):
    """
    📂 Load a document (.txt, .pdf, .docx) and prepare it for QA.
    """
    if not file.exists():
        typer.echo("❌ File not found.")
        raise typer.Exit()

    text = extract_text(str(file))
    context = ChatbotContext()
    context.load_text(text)
    save_context(context)
    typer.echo("✅ Document loaded and context saved.")

@app.command()
def ask(question: str):
    """
    ❓ Ask a question about the loaded document.
    """
    context = load_context()
    if not context or not context.text:
        typer.echo("⚠️ Please load a document first using: python clichat.py load <file>")
        raise typer.Exit()

    greet = greeting(question)
    response = greet if greet else respond(context, question)
    typer.echo(f"🤖 ROBO: {response}")

@app.command()
def summarize_doc():
    """
    📝 Show a summary of the loaded document.
    """
    context = load_context()
    if not context or not context.text:
        typer.echo("⚠️ Please load a document first.")
        raise typer.Exit()

    typer.echo("📄 Summary:\n")
    typer.echo(summarize(context.text))

@app.command()
def chat():
    """
    💬 Start an interactive chat with ROBO.
    """
    context = load_context()
    if not context or not context.text:
        typer.echo("⚠️ Please load a document first.")
        raise typer.Exit()

    typer.echo("💬 Start chatting with Robo! Type 'exit' to quit.")
    while True:
        user_input = input("YOU: ")
        if user_input.lower() in ['exit', 'quit']:
            typer.echo("ROBO: Goodbye!")
            break
        greet = greeting(user_input)
        response = greet if greet else respond(context, user_input)
        typer.echo(f"ROBO: {response}")

if __name__ == "__main__":
    app()
