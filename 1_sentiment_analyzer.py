"""
Sentiment Analyzer
-------------------
Domain: AI/ML
OOP: Preprocessor, Model (NaiveBayesModel), Analyzer
DSA/Concepts: tokenization, dictionary frequency maps, Naive Bayes probability

A simple bag-of-words Naive Bayes sentiment classifier built from scratch
(no external ML libraries required).
"""

import re
import math
from collections import defaultdict


class Preprocessor:
    """Handles text cleaning and tokenization."""

    STOPWORDS = {
        "the", "a", "an", "is", "it", "to", "of", "and", "in", "on",
        "this", "that", "was", "were", "be", "been", "i", "we", "you"
    }

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text

    def tokenize(self, text: str) -> list:
        text = self.clean(text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS]
        return tokens


class NaiveBayesModel:
    """Multinomial Naive Bayes implemented with frequency dictionaries."""

    def __init__(self):
        # class -> {word: count}
        self.word_counts = defaultdict(lambda: defaultdict(int))
        # class -> total word count
        self.class_totals = defaultdict(int)
        # class -> number of documents
        self.class_doc_counts = defaultdict(int)
        self.vocabulary = set()
        self.classes = set()
        self.total_docs = 0

    def train(self, tokenized_docs: list, labels: list):
        self.total_docs = len(tokenized_docs)
        for tokens, label in zip(tokenized_docs, labels):
            self.classes.add(label)
            self.class_doc_counts[label] += 1
            for word in tokens:
                self.word_counts[label][word] += 1
                self.class_totals[label] += 1
                self.vocabulary.add(word)

    def _log_likelihood(self, word: str, label: str) -> float:
        # Laplace (add-one) smoothing
        count = self.word_counts[label].get(word, 0)
        return math.log(
            (count + 1) / (self.class_totals[label] + len(self.vocabulary))
        )

    def predict_proba(self, tokens: list) -> dict:
        scores = {}
        for label in self.classes:
            log_prior = math.log(self.class_doc_counts[label] / self.total_docs)
            log_prob = log_prior
            for word in tokens:
                if word in self.vocabulary:
                    log_prob += self._log_likelihood(word, label)
            scores[label] = log_prob

        # convert log-scores to normalized probabilities
        max_log = max(scores.values())
        exp_scores = {k: math.exp(v - max_log) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: v / total for k, v in exp_scores.items()}

    def predict(self, tokens: list) -> str:
        probs = self.predict_proba(tokens)
        return max(probs, key=probs.get)


class Analyzer:
    """High level facade tying Preprocessor + Model together."""

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = NaiveBayesModel()

    def fit(self, texts: list, labels: list):
        tokenized = [self.preprocessor.tokenize(t) for t in texts]
        self.model.train(tokenized, labels)

    def analyze(self, text: str) -> dict:
        tokens = self.preprocessor.tokenize(text)
        label = self.model.predict(tokens)
        probs = self.model.predict_proba(tokens)
        return {"text": text, "sentiment": label, "confidence": round(probs[label], 4)}


def main():
    train_texts = [
        "I love this product, it works great",
        "Absolutely fantastic experience, will buy again",
        "Best purchase I have made this year",
        "This is amazing and wonderful",
        "I hate this, it broke immediately",
        "Terrible quality, waste of money",
        "Worst experience ever, very disappointed",
        "This is awful and frustrating",
    ]
    train_labels = ["positive", "positive", "positive", "positive",
                     "negative", "negative", "negative", "negative"]

    analyzer = Analyzer()
    analyzer.fit(train_texts, train_labels)

    test_sentences = [
        "I really love how great this works",
        "This product is terrible and broke",
        "Not bad, pretty decent actually",
    ]

    print("=== Sentiment Analyzer Demo ===")
    for sentence in test_sentences:
        result = analyzer.analyze(sentence)
        print(f"Text: {result['text']}")
        print(f"  -> Sentiment: {result['sentiment']} (confidence: {result['confidence']})")


if __name__ == "__main__":
    main()
