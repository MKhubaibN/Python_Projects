"""
Network Traffic Classifier
-----------------------------
Domain: AI/ML + Cybersecurity
OOP: Flow, Feature, Classifier
DSA/Concepts: K-Nearest Neighbors, feature vectors, Euclidean distance, heaps/sorting
"""

import math
from collections import Counter


class Feature:
    """Defines a single feature with optional normalization bounds."""

    def __init__(self, name: str, min_val: float, max_val: float):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, value: float) -> float:
        if self.max_val == self.min_val:
            return 0.0
        return (value - self.min_val) / (self.max_val - self.min_val)


class Flow:
    """Represents a single network flow with raw feature values + label."""

    def __init__(self, packet_count: int, byte_count: int, duration: float,
                 avg_packet_size: float, label: str = None):
        self.packet_count = packet_count
        self.byte_count = byte_count
        self.duration = duration
        self.avg_packet_size = avg_packet_size
        self.label = label

    def to_vector(self) -> list:
        return [self.packet_count, self.byte_count, self.duration, self.avg_packet_size]

    def __repr__(self):
        return f"Flow(pkts={self.packet_count}, bytes={self.byte_count}, label={self.label})"


class Classifier:
    """K-Nearest Neighbors classifier for network flow classification."""

    FEATURE_DEFS = [
        Feature("packet_count", 0, 10000),
        Feature("byte_count", 0, 5_000_000),
        Feature("duration", 0, 300),
        Feature("avg_packet_size", 0, 1500),
    ]

    def __init__(self, k: int = 3):
        self.k = k
        self.training_flows = []
        self.training_vectors = []

    def _normalize_vector(self, vector: list) -> list:
        return [f.normalize(v) for f, v in zip(self.FEATURE_DEFS, vector)]

    def fit(self, flows: list):
        self.training_flows = flows
        self.training_vectors = [
            self._normalize_vector(flow.to_vector()) for flow in flows
        ]

    @staticmethod
    def _euclidean_distance(v1: list, v2: list) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def _k_nearest(self, vector: list) -> list:
        distances = [
            (self._euclidean_distance(vector, train_vec), flow)
            for train_vec, flow in zip(self.training_vectors, self.training_flows)
        ]
        distances.sort(key=lambda pair: pair[0])
        return distances[: self.k]

    def predict(self, flow: Flow) -> str:
        vector = self._normalize_vector(flow.to_vector())
        neighbors = self._k_nearest(vector)
        labels = [n[1].label for n in neighbors]
        vote_counts = Counter(labels)
        return vote_counts.most_common(1)[0][0]

    def predict_with_confidence(self, flow: Flow) -> dict:
        vector = self._normalize_vector(flow.to_vector())
        neighbors = self._k_nearest(vector)
        labels = [n[1].label for n in neighbors]
        vote_counts = Counter(labels)
        top_label, top_count = vote_counts.most_common(1)[0]
        confidence = top_count / self.k
        return {"label": top_label, "confidence": round(confidence, 2),
                "neighbors": [(round(d, 3), f.label) for d, f in neighbors]}


def build_training_set() -> list:
    """Synthetic labeled flows: normal, dns, port_scan, ddos."""
    return [
        # Normal web browsing traffic
        Flow(50, 60000, 5.0, 1200, label="normal"),
        Flow(45, 55000, 4.5, 1222, label="normal"),
        Flow(60, 72000, 6.0, 1200, label="normal"),
        # DNS queries (small, fast)
        Flow(2, 150, 0.1, 75, label="dns"),
        Flow(3, 200, 0.15, 66, label="dns"),
        Flow(2, 140, 0.12, 70, label="dns"),
        # Port scan (many tiny packets, short duration)
        Flow(500, 30000, 1.0, 60, label="port_scan"),
        Flow(600, 36000, 1.2, 60, label="port_scan"),
        Flow(550, 33000, 1.1, 60, label="port_scan"),
        # DDoS (huge packet/byte counts)
        Flow(9000, 4_500_000, 10.0, 500, label="ddos"),
        Flow(8500, 4_200_000, 9.5, 494, label="ddos"),
        Flow(9500, 4_800_000, 11.0, 505, label="ddos"),
    ]


def main():
    classifier = Classifier(k=3)
    classifier.fit(build_training_set())

    test_flows = [
        Flow(48, 58000, 5.2, 1208),    # should be normal
        Flow(2, 160, 0.13, 80),        # should be dns
        Flow(580, 34800, 1.15, 60),    # should be port_scan
        Flow(9100, 4_600_000, 10.5, 505),  # should be ddos
    ]

    print("=== Network Traffic Classifier (KNN) Demo ===\n")
    for flow in test_flows:
        result = classifier.predict_with_confidence(flow)
        print(f"Flow: {flow}")
        print(f"  -> Predicted: {result['label']} (confidence: {result['confidence']})")
        print(f"  Nearest neighbors: {result['neighbors']}\n")


if __name__ == "__main__":
    main()
