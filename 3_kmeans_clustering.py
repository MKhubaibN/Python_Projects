"""
K-Means Clustering
--------------------
Domain: AI/ML
OOP: Dataset, Centroid, Cluster, KMeans
DSA/Concepts: Euclidean distance, iterative convergence, lists of vectors
"""

import math
import random


class Dataset:
    """Wraps a list of n-dimensional points."""

    def __init__(self, points: list):
        self.points = points  # list of tuples/lists

    def __len__(self):
        return len(self.points)

    def dimensions(self) -> int:
        return len(self.points[0]) if self.points else 0


class Centroid:
    """Represents the center of a cluster."""

    def __init__(self, coordinates: list):
        self.coordinates = list(coordinates)

    def distance_to(self, point: list) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.coordinates, point)))

    def recompute(self, points: list):
        if not points:
            return
        dims = len(points[0])
        self.coordinates = [
            sum(p[d] for p in points) / len(points) for d in range(dims)
        ]

    def __repr__(self):
        return f"Centroid({[round(c, 3) for c in self.coordinates]})"


class Cluster:
    """A centroid plus the points currently assigned to it."""

    def __init__(self, centroid: Centroid):
        self.centroid = centroid
        self.points = []

    def add_point(self, point: list):
        self.points.append(point)

    def clear(self):
        self.points = []

    def update_centroid(self):
        self.centroid.recompute(self.points)


class KMeans:
    """Iterative K-Means clustering implementation."""

    def __init__(self, k: int, max_iterations: int = 100, tolerance: float = 1e-4, seed: int = 42):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.clusters = []
        random.seed(seed)

    def _init_centroids(self, dataset: Dataset):
        initial_points = random.sample(dataset.points, self.k)
        self.clusters = [Cluster(Centroid(p)) for p in initial_points]

    def _assign_points(self, dataset: Dataset):
        for cluster in self.clusters:
            cluster.clear()

        for point in dataset.points:
            distances = [c.centroid.distance_to(point) for c in self.clusters]
            nearest_index = distances.index(min(distances))
            self.clusters[nearest_index].add_point(point)

    def _has_converged(self, old_centroids: list) -> bool:
        for old, cluster in zip(old_centroids, self.clusters):
            if cluster.centroid.distance_to(old) > self.tolerance:
                return False
        return True

    def fit(self, dataset: Dataset):
        self._init_centroids(dataset)

        for iteration in range(self.max_iterations):
            self._assign_points(dataset)
            old_centroids = [c.centroid.coordinates[:] for c in self.clusters]

            for cluster in self.clusters:
                cluster.update_centroid()

            if self._has_converged(old_centroids):
                print(f"Converged after {iteration + 1} iterations.")
                break

        return self.clusters

    def predict(self, point: list) -> int:
        distances = [c.centroid.distance_to(point) for c in self.clusters]
        return distances.index(min(distances))


def main():
    # Synthetic 2D dataset with 3 natural groupings
    points = [
        (1, 1), (1.5, 2), (2, 1.5), (1.2, 1.8),
        (8, 8), (8.5, 9), (9, 8.2), (8.7, 7.9),
        (1, 9), (1.4, 8.6), (0.8, 9.5), (1.6, 9.2),
    ]

    dataset = Dataset(points)
    model = KMeans(k=3)
    clusters = model.fit(dataset)

    print("\n=== K-Means Clustering Demo ===")
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i + 1} - {cluster.centroid}")
        print(f"  Points: {cluster.points}")

    test_point = [1.3, 1.6]
    cluster_id = model.predict(test_point)
    print(f"\nNew point {test_point} assigned to Cluster {cluster_id + 1}")


if __name__ == "__main__":
    main()
