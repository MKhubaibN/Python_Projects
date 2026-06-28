"""
Log File Anomaly Detector
----------------------------
Domain: Cybersecurity
OOP: Parser, Detector, Alert
DSA/Concepts: sliding window, frequency maps, z-score statistics
"""

import re
import statistics
from collections import defaultdict, deque
from datetime import datetime


class LogEntry:
    """Represents one parsed log line."""

    def __init__(self, timestamp: datetime, ip: str, status_code: int, raw: str):
        self.timestamp = timestamp
        self.ip = ip
        self.status_code = status_code
        self.raw = raw

    def __repr__(self):
        return f"LogEntry({self.timestamp}, {self.ip}, {self.status_code})"


class Parser:
    """Parses raw log lines into LogEntry objects.

    Expected format: "2024-01-01 12:00:01 192.168.1.10 GET /login 401"
    """

    LOG_PATTERN = re.compile(
        r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
        r"(?P<ip>\d+\.\d+\.\d+\.\d+)\s+\S+\s+\S+\s+(?P<status>\d{3})"
    )

    def parse_line(self, line: str) -> LogEntry:
        match = self.LOG_PATTERN.search(line)
        if not match:
            return None
        dt = datetime.strptime(
            f"{match.group('date')} {match.group('time')}", "%Y-%m-%d %H:%M:%S"
        )
        return LogEntry(dt, match.group("ip"), int(match.group("status")), line)

    def parse_lines(self, lines: list) -> list:
        entries = []
        for line in lines:
            entry = self.parse_line(line)
            if entry:
                entries.append(entry)
        return entries


class Alert:
    """A single anomaly alert."""

    def __init__(self, ip: str, reason: str, severity: str, evidence: dict):
        self.ip = ip
        self.reason = reason
        self.severity = severity
        self.evidence = evidence

    def __repr__(self):
        return f"[{self.severity}] {self.ip} -> {self.reason} | {self.evidence}"


class Detector:
    """Detects anomalies using sliding windows and z-score frequency analysis."""

    def __init__(self, window_size: int = 10, request_threshold: int = 5,
                 failed_login_threshold: int = 3, zscore_threshold: float = 2.0):
        self.window_size = window_size
        self.request_threshold = request_threshold
        self.failed_login_threshold = failed_login_threshold
        self.zscore_threshold = zscore_threshold

    def detect_request_floods(self, entries: list) -> list:
        """Sliding window over time-ordered entries per IP to flag bursts."""
        alerts = []
        ip_windows = defaultdict(deque)

        for entry in entries:
            window = ip_windows[entry.ip]
            window.append(entry.timestamp)

            # slide window: drop entries older than window_size seconds
            while window and (entry.timestamp - window[0]).total_seconds() > self.window_size:
                window.popleft()

            if len(window) >= self.request_threshold:
                alerts.append(
                    Alert(
                        entry.ip,
                        "Request flood detected",
                        "HIGH",
                        {"requests_in_window": len(window), "window_seconds": self.window_size},
                    )
                )
        return alerts

    def detect_failed_logins(self, entries: list) -> list:
        """Frequency map of failed (4xx) status codes per IP."""
        alerts = []
        failure_counts = defaultdict(int)

        for entry in entries:
            if 400 <= entry.status_code < 500:
                failure_counts[entry.ip] += 1

        for ip, count in failure_counts.items():
            if count >= self.failed_login_threshold:
                alerts.append(
                    Alert(ip, "Repeated failed requests / possible brute force",
                          "MEDIUM", {"failed_attempts": count})
                )
        return alerts

    def detect_statistical_outliers(self, entries: list) -> list:
        """Z-score analysis on per-IP request counts to find outlier IPs."""
        alerts = []
        ip_counts = defaultdict(int)
        for entry in entries:
            ip_counts[entry.ip] += 1

        counts = list(ip_counts.values())
        if len(counts) < 2:
            return alerts

        mean = statistics.mean(counts)
        stdev = statistics.stdev(counts)
        if stdev == 0:
            return alerts

        for ip, count in ip_counts.items():
            z = (count - mean) / stdev
            if z >= self.zscore_threshold:
                alerts.append(
                    Alert(ip, "Statistical traffic outlier", "MEDIUM",
                          {"request_count": count, "z_score": round(z, 2)})
                )
        return alerts

    def run(self, entries: list) -> list:
        alerts = []
        alerts.extend(self.detect_request_floods(entries))
        alerts.extend(self.detect_failed_logins(entries))
        alerts.extend(self.detect_statistical_outliers(entries))
        return alerts


def generate_sample_log() -> list:
    """Generates synthetic log lines simulating normal + suspicious traffic."""
    lines = []
    base = "2024-01-01 12:00:{:02d} {} GET /page {}"

    # normal traffic
    for i in range(0, 20, 2):
        lines.append(base.format(i, "10.0.0.5", 200))

    # flood from a single IP within seconds
    for i in range(20, 30):
        lines.append(base.format(i, "192.168.1.50", 200))

    # brute-force failed login attempts
    for i in range(0, 5):
        lines.append(base.format(30 + i, "203.0.113.9", 401))

    return lines


def main():
    parser = Parser()
    detector = Detector()

    raw_lines = generate_sample_log()
    entries = parser.parse_lines(raw_lines)

    print(f"=== Log File Anomaly Detector Demo ===")
    print(f"Parsed {len(entries)} log entries.\n")

    alerts = detector.run(entries)
    if not alerts:
        print("No anomalies detected.")
    for alert in alerts:
        print(alert)


if __name__ == "__main__":
    main()
