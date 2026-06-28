"""
Hash Cracker Simulator
-------------------------
Domain: Cybersecurity
OOP: HashEngine, Dictionary, Cracker
DSA/Concepts: hash maps (precomputed lookup table), brute force with itertools,
              MD5/SHA via hashlib

NOTE: Educational/demo tool only - operates on a small local wordlist
and a limited brute-force charset, never against real-world systems.
"""

import hashlib
import itertools
import time


class HashEngine:
    """Wraps hashlib to compute hashes with a chosen algorithm."""

    SUPPORTED_ALGOS = {"md5", "sha1", "sha256"}

    def __init__(self, algorithm: str = "sha256"):
        if algorithm not in self.SUPPORTED_ALGOS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        self.algorithm = algorithm

    def hash(self, text: str) -> str:
        h = hashlib.new(self.algorithm)
        h.update(text.encode("utf-8"))
        return h.hexdigest()


class Dictionary:
    """Holds a wordlist and exposes a precomputed hash -> word lookup map."""

    def __init__(self, words: list, engine: HashEngine):
        self.words = words
        self.engine = engine
        self.hash_lookup = self._build_lookup()

    def _build_lookup(self) -> dict:
        # Hash map: hashed_value -> original word (O(1) average lookup)
        return {self.engine.hash(word): word for word in self.words}

    def lookup(self, target_hash: str):
        return self.hash_lookup.get(target_hash)


class Cracker:
    """Attempts to recover plaintext from a hash via dictionary then brute force."""

    def __init__(self, engine: HashEngine, dictionary: Dictionary,
                 brute_force_charset: str = "abcdefghijklmnopqrstuvwxyz0123456789",
                 max_brute_force_length: int = 4):
        self.engine = engine
        self.dictionary = dictionary
        self.charset = brute_force_charset
        self.max_length = max_brute_force_length

    def dictionary_attack(self, target_hash: str):
        return self.dictionary.lookup(target_hash)

    def brute_force_attack(self, target_hash: str):
        """Tries all combinations up to max_length using itertools.product."""
        for length in range(1, self.max_length + 1):
            for combo in itertools.product(self.charset, repeat=length):
                candidate = "".join(combo)
                if self.engine.hash(candidate) == target_hash:
                    return candidate
        return None

    def crack(self, target_hash: str) -> dict:
        start = time.time()

        result = self.dictionary_attack(target_hash)
        method = "dictionary"

        if result is None:
            result = self.brute_force_attack(target_hash)
            method = "brute_force" if result else "failed"

        elapsed = round(time.time() - start, 4)
        return {
            "target_hash": target_hash,
            "cracked": result,
            "method": method,
            "time_seconds": elapsed,
        }


def main():
    engine = HashEngine(algorithm="sha256")

    common_words = ["password", "admin", "letmein", "qwerty", "sunshine",
                     "dragon", "football", "monkey", "shadow", "master"]
    dictionary = Dictionary(common_words, engine)

    cracker = Cracker(engine, dictionary, max_brute_force_length=3)

    print("=== Hash Cracker Simulator Demo ===\n")

    # Case 1: hash exists in dictionary
    target1 = engine.hash("dragon")
    result1 = cracker.crack(target1)
    print(f"Target hash: {target1[:20]}...")
    print(f"Result: {result1}\n")

    # Case 2: short password not in dictionary - falls back to brute force
    target2 = engine.hash("ab1")
    result2 = cracker.crack(target2)
    print(f"Target hash: {target2[:20]}...")
    print(f"Result: {result2}\n")

    # Case 3: unrecoverable within limits
    target3 = engine.hash("zzzz9!@#")
    result3 = cracker.crack(target3)
    print(f"Target hash: {target3[:20]}...")
    print(f"Result: {result3}")


if __name__ == "__main__":
    main()
