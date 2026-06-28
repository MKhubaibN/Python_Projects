"""
Spam Email Classifier
------------------------
Domain: AI/ML
OOP: EmailParser, Vocabulary, Classifier
DSA/Concepts: TF-IDF, Naive Bayes, confusion matrix, hash maps
"""

import re
import math
from collections import defaultdict, Counter


class EmailParser:
    """Cleans and tokenizes raw email text."""

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " URL ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text

    def tokenize(self, text: str) -> list:
        return self.clean(text).split()


class Vocabulary:
    """Builds vocabulary and TF-IDF weighting from a corpus of tokenized docs."""

    def __init__(self):
        self.doc_freq = defaultdict(int)   # word -> number of docs containing it
        self.total_docs = 0
        self.words = set()

    def build(self, tokenized_docs: list):
        self.total_docs = len(tokenized_docs)
        for tokens in tokenized_docs:
            unique_words = set(tokens)
            self.words.update(unique_words)
            for word in unique_words:
                self.doc_freq[word] += 1

    def idf(self, word: str) -> float:
        df = self.doc_freq.get(word, 0)
        # smoothed idf to avoid division by zero / log(0)
        return math.log((1 + self.total_docs) / (1 + df)) + 1

    def tfidf_vector(self, tokens: list) -> dict:
        term_freq = Counter(tokens)
        total_terms = len(tokens) or 1
        return {
            word: (count / total_terms) * self.idf(word)
            for word, count in term_freq.items()
        }


class Classifier:
    """Naive Bayes classifier operating on TF-IDF-weighted word presence."""

    def __init__(self):
        self.parser = EmailParser()
        self.vocabulary = Vocabulary()
        self.class_word_weight = defaultdict(lambda: defaultdict(float))
        self.class_totals = defaultdict(float)
        self.class_doc_counts = defaultdict(int)
        self.classes = set()
        self.total_docs = 0

    def fit(self, emails: list, labels: list):
        tokenized_docs = [self.parser.tokenize(e) for e in emails]
        self.vocabulary.build(tokenized_docs)
        self.total_docs = len(emails)

        for tokens, label in zip(tokenized_docs, labels):
            self.classes.add(label)
            self.class_doc_counts[label] += 1
            tfidf = self.vocabulary.tfidf_vector(tokens)
            for word, weight in tfidf.items():
                self.class_word_weight[label][word] += weight
                self.class_totals[label] += weight

    def _log_likelihood(self, word: str, weight: float, label: str) -> float:
        word_weight = self.class_word_weight[label].get(word, 0.0)
        vocab_size = len(self.vocabulary.words)
        # Laplace smoothing on weighted counts
        prob = (word_weight + 1) / (self.class_totals[label] + vocab_size)
        return weight * math.log(prob)

    def predict_proba(self, email_text: str) -> dict:
        tokens = self.parser.tokenize(email_text)
        tfidf = self.vocabulary.tfidf_vector(tokens)

        scores = {}
        for label in self.classes:
            log_prior = math.log(self.class_doc_counts[label] / self.total_docs)
            log_prob = log_prior
            for word, weight in tfidf.items():
                log_prob += self._log_likelihood(word, weight, label)
            scores[label] = log_prob

        max_log = max(scores.values())
        exp_scores = {k: math.exp(v - max_log) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: v / total for k, v in exp_scores.items()}

    def predict(self, email_text: str) -> str:
        probs = self.predict_proba(email_text)
        return max(probs, key=probs.get)


class ConfusionMatrix:
    """Computes and displays a confusion matrix for evaluation."""

    def __init__(self, y_true: list, y_pred: list, labels: list):
        self.labels = labels
        self.matrix = {a: {p: 0 for p in labels} for a in labels}
        for actual, predicted in zip(y_true, y_pred):
            self.matrix[actual][predicted] += 1

    def accuracy(self, y_true: list, y_pred: list) -> float:
        correct = sum(1 for a, p in zip(y_true, y_pred) if a == p)
        return correct / len(y_true)

    def display(self):
        header = "Actual\\Pred".ljust(12) + "".join(l.ljust(10) for l in self.labels)
        print(header)
        for actual in self.labels:
            row = actual.ljust(12) + "".join(
                str(self.matrix[actual][p]).ljust(10) for p in self.labels
            )
            print(row)


def main():
    emails = [
        "Win a free iPhone now click this link http://scam.com",
        "Congratulations you have won a lottery prize claim now",
        "Limited offer buy cheap meds online discount",
        "Free money click here urgent winner selected",
        "Hi John can we reschedule our meeting to Friday",
        "Please find attached the quarterly report for review",
        "Reminder your dentist appointment is tomorrow at 10am",
        "Let's catch up for coffee sometime this week",
    ]
    labels = ["spam", "spam", "spam", "spam", "ham", "ham", "ham", "ham"]

    classifier = Classifier()
    classifier.fit(emails, labels)

    test_emails = [
        "Click here to claim your free prize now",
        "Can we move our meeting to next Monday",
        "Urgent winner click link to claim cash reward",
        "Attached is the document you requested",
    ]
    expected = ["spam", "ham", "spam", "ham"]

    predictions = [classifier.predict(e) for e in test_emails]

    print("=== Spam Email Classifier Demo ===\n")
    for email, pred, exp in zip(test_emails, predictions, expected):
        probs = classifier.predict_proba(email)
        print(f"Email: {email}")
        print(f"  Predicted: {pred} (expected: {exp}) | Confidence: {round(probs[pred], 3)}\n")

    cm = ConfusionMatrix(expected, predictions, labels=["spam", "ham"])
    print("Confusion Matrix:")
    cm.display()
    print(f"\nAccuracy: {cm.accuracy(expected, predictions) * 100:.1f}%")


if __name__ == "__main__":
    main()
