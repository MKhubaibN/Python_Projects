"""
Recommendation Engine
------------------------
Domain: AI/ML
OOP: User, Item, Engine
DSA/Concepts: cosine similarity, collaborative filtering, sparse hash maps
"""

import math
from collections import defaultdict


class Item:
    """Represents a recommendable item (e.g. movie, product)."""

    def __init__(self, item_id: str, name: str):
        self.item_id = item_id
        self.name = name

    def __repr__(self):
        return f"Item({self.item_id}: {self.name})"


class User:
    """Represents a user and their ratings: item_id -> rating."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.ratings = {}  # item_id -> rating (e.g. 1-5)

    def rate(self, item_id: str, rating: float):
        self.ratings[item_id] = rating

    def __repr__(self):
        return f"User({self.user_id}, {len(self.ratings)} ratings)"


class Engine:
    """User-based collaborative filtering recommendation engine."""

    def __init__(self):
        self.users = {}   # user_id -> User
        self.items = {}   # item_id -> Item

    def add_user(self, user: User):
        self.users[user.user_id] = user

    def add_item(self, item: Item):
        self.items[item.item_id] = item

    @staticmethod
    def _cosine_similarity(ratings_a: dict, ratings_b: dict) -> float:
        """Cosine similarity over the items both users have rated in common."""
        common_items = set(ratings_a.keys()) & set(ratings_b.keys())
        if not common_items:
            return 0.0

        dot_product = sum(ratings_a[i] * ratings_b[i] for i in common_items)
        norm_a = math.sqrt(sum(v ** 2 for v in ratings_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in ratings_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def find_similar_users(self, target_user_id: str, top_n: int = 3) -> list:
        target = self.users[target_user_id]
        similarities = []

        for user_id, user in self.users.items():
            if user_id == target_user_id:
                continue
            sim = self._cosine_similarity(target.ratings, user.ratings)
            if sim > 0:
                similarities.append((user_id, sim))

        similarities.sort(key=lambda pair: pair[1], reverse=True)
        return similarities[:top_n]

    def recommend(self, target_user_id: str, top_n: int = 5, neighbor_count: int = 3) -> list:
        """Predicts ratings for unrated items using weighted neighbor ratings."""
        target = self.users[target_user_id]
        neighbors = self.find_similar_users(target_user_id, top_n=neighbor_count)

        # weighted_sum[item] -> accumulated (similarity * rating)
        weighted_sum = defaultdict(float)
        similarity_sum = defaultdict(float)

        for neighbor_id, similarity in neighbors:
            neighbor = self.users[neighbor_id]
            for item_id, rating in neighbor.ratings.items():
                if item_id in target.ratings:
                    continue  # skip items target user already rated
                weighted_sum[item_id] += similarity * rating
                similarity_sum[item_id] += similarity

        predicted_scores = {
            item_id: weighted_sum[item_id] / similarity_sum[item_id]
            for item_id in weighted_sum
            if similarity_sum[item_id] > 0
        }

        ranked = sorted(predicted_scores.items(), key=lambda pair: pair[1], reverse=True)
        return [
            (self.items[item_id].name, round(score, 2))
            for item_id, score in ranked[:top_n]
        ]


def build_demo_engine() -> Engine:
    engine = Engine()

    items = [
        Item("m1", "Inception"),
        Item("m2", "The Matrix"),
        Item("m3", "Interstellar"),
        Item("m4", "The Notebook"),
        Item("m5", "Titanic"),
        Item("m6", "Mad Max: Fury Road"),
    ]
    for item in items:
        engine.add_item(item)

    ratings_data = {
        "alice": {"m1": 5, "m2": 5, "m3": 4, "m4": 1},
        "bob":   {"m1": 5, "m2": 4, "m3": 5, "m6": 4},
        "carol": {"m4": 5, "m5": 5, "m1": 2},
        "dave":  {"m1": 4, "m2": 5, "m6": 5},
        "eve":   {"m4": 4, "m5": 4, "m3": 2},
    }

    for user_id, ratings in ratings_data.items():
        user = User(user_id)
        for item_id, rating in ratings.items():
            user.rate(item_id, rating)
        engine.add_user(user)

    return engine


def main():
    engine = build_demo_engine()

    print("=== Recommendation Engine Demo (Collaborative Filtering) ===\n")

    for target_user_id in ["alice", "carol"]:
        print(f"Target user: {target_user_id}")
        similar = engine.find_similar_users(target_user_id)
        print(f"  Most similar users: {similar}")

        recommendations = engine.recommend(target_user_id, top_n=3)
        print(f"  Recommended items: {recommendations}\n")


if __name__ == "__main__":
    main()
