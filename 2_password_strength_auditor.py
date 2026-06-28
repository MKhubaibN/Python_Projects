"""
Password Strength Auditor
---------------------------
Domain: Cybersecurity
OOP: Rule, Auditor, Report
DSA/Concepts: regex, hash maps for scoring, weighted scoring algorithm
"""

import re
import hashlib
from collections import Counter


class Rule:
    """A single scoring rule applied to a password."""

    def __init__(self, name: str, weight: int, test_func, message: str):
        self.name = name
        self.weight = weight
        self.test_func = test_func
        self.message = message

    def evaluate(self, password: str):
        passed = self.test_func(password)
        score = self.weight if passed else 0
        feedback = None if passed else self.message
        return passed, score, feedback


class Report:
    """Holds the results of an audit and renders them."""

    def __init__(self, password: str, max_score: int):
        self.password = password
        self.max_score = max_score
        self.score = 0
        self.feedback = []
        self.rule_results = {}

    def add_result(self, rule_name: str, passed: bool, score: int, feedback: str):
        self.rule_results[rule_name] = passed
        self.score += score
        if feedback:
            self.feedback.append(feedback)

    @property
    def percentage(self) -> float:
        return round((self.score / self.max_score) * 100, 1) if self.max_score else 0

    @property
    def strength_label(self) -> str:
        pct = self.percentage
        if pct >= 85:
            return "Very Strong"
        elif pct >= 65:
            return "Strong"
        elif pct >= 45:
            return "Moderate"
        elif pct >= 25:
            return "Weak"
        return "Very Weak"

    def summary(self) -> str:
        lines = [
            f"Password: {'*' * len(self.password)}",
            f"Score: {self.score}/{self.max_score} ({self.percentage}%)",
            f"Strength: {self.strength_label}",
        ]
        if self.feedback:
            lines.append("Suggestions:")
            for f in self.feedback:
                lines.append(f"  - {f}")
        else:
            lines.append("No issues found. Great password!")
        return "\n".join(lines)


class Auditor:
    """Builds rules and evaluates passwords against them."""

    COMMON_PASSWORDS = {
        "password", "123456", "12345678", "qwerty", "abc123",
        "letmein", "111111", "admin", "welcome", "iloveyou"
    }

    def __init__(self):
        self.rules = self._build_rules()
        self.max_score = sum(r.weight for r in self.rules)
        # frequency map used for repeated-character detection
        self._char_freq_cache = {}

    def _build_rules(self) -> list:
        return [
            Rule("length_8", 20, lambda p: len(p) >= 8,
                 "Use at least 8 characters."),
            Rule("length_12", 10, lambda p: len(p) >= 12,
                 "Consider 12+ characters for stronger security."),
            Rule("uppercase", 15, lambda p: bool(re.search(r"[A-Z]", p)),
                 "Add at least one uppercase letter."),
            Rule("lowercase", 15, lambda p: bool(re.search(r"[a-z]", p)),
                 "Add at least one lowercase letter."),
            Rule("digit", 15, lambda p: bool(re.search(r"\d", p)),
                 "Add at least one digit."),
            Rule("special", 15, lambda p: bool(re.search(r"[!@#$%^&*(),.?\":{}|<>_\-]", p)),
                 "Add at least one special character (e.g. !@#$%)."),
            Rule("not_common", 10, lambda p: p.lower() not in self.COMMON_PASSWORDS,
                 "Avoid commonly used passwords."),
            Rule("no_repeats", 10, lambda p: not self._has_excessive_repeats(p),
                 "Avoid long runs of the same repeated character."),
        ]

    def _has_excessive_repeats(self, password: str) -> bool:
        # uses a frequency map (hash map) to detect repeated-char runs
        freq = Counter(password)
        self._char_freq_cache[password] = freq
        max_run, current_run = 1, 1
        for i in range(1, len(password)):
            if password[i] == password[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run >= 4

    def audit(self, password: str) -> Report:
        report = Report(password, self.max_score)
        for rule in self.rules:
            passed, score, feedback = rule.evaluate(password)
            report.add_result(rule.name, passed, score, feedback)
        return report

    @staticmethod
    def hash_password(password: str, algo: str = "sha256") -> str:
        """Returns a hex digest of the password using hashlib (for storage demo)."""
        h = hashlib.new(algo)
        h.update(password.encode("utf-8"))
        return h.hexdigest()


def main():
    auditor = Auditor()
    test_passwords = [
        "password",
        "Summer2023",
        "Tr0ub4dor&3!Lengthy",
        "aaaaaaaa",
        "C0mpl3x!Pass#2024",
    ]

    print("=== Password Strength Auditor Demo ===\n")
    for pwd in test_passwords:
        report = auditor.audit(pwd)
        print(report.summary())
        print(f"SHA-256: {auditor.hash_password(pwd)[:24]}...")
        print("-" * 50)


if __name__ == "__main__":
    main()
