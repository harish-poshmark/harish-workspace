#!/usr/bin/env python3
"""Generate sentiment summary for Reddit and X chatter about Poshmark consignment."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REPORT_DIR = REPO_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "have",
    "your",
    "about",
    "into",
    "when",
    "they",
    "been",
    "from",
    "into",
    "been",
    "will",
    "just",
    "them",
    "then",
    "here",
    "there",
    "what",
    "need",
    "want",
    "consignment",
    "poshmark",
    "bag",
    "bags",
}

TOPIC_KEYWORDS = (
    "consignor",
    "closet partner",
    "partner",
    "consignment bag",
    "bag",
    "bags",
    "sell for",
    "consignment network",
    "consignment program",
    "sent my bag",
    "received my bag",
)

WINDOW_DAYS = 60


def load_json(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return json.loads(path.read_text())


def strip_markdown_links(text: str) -> str:
    without_links = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    without_hash = re.sub(r"#[A-Za-z0-9_]+", "", without_links)
    cleaned = re.sub(r"\s+", " ", without_hash).strip()
    return cleaned


def add_sentiment(posts: List[Dict]) -> List[Dict]:
    analyzer = SentimentIntensityAnalyzer()
    enriched = []
    for post in posts:
        text = strip_markdown_links(post.get("text", ""))
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            label = "positive"
        elif score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        enriched.append({**post, "clean_text": text, "sentiment": score, "sentiment_label": label})
    return enriched


def parse_timestamp(post: Dict) -> Optional[datetime]:
    raw = post.get("timestamp")
    if raw:
        cleaned = raw.replace("·", "").replace("  ", " ").strip()
        for fmt in ("%b %d, %Y %I:%M %p %Z", "%b %d, %Y %I:%M %p"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
    retrieved = post.get("retrieved_at")
    if retrieved:
        try:
            if retrieved.endswith("Z"):
                retrieved = retrieved.replace("Z", "+00:00")
            return datetime.fromisoformat(retrieved).astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def filter_recent(posts: List[Dict], days: int = WINDOW_DAYS) -> List[Dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for post in posts:
        ts = parse_timestamp(post)
        if ts is None:
            continue
        if ts >= cutoff:
            filtered.append(post)
    return filtered


def matches_topic(post: Dict) -> bool:
    text = post.get("clean_text", "").lower()
    if not text:
        return False
    has_keyword = any(keyword in text for keyword in TOPIC_KEYWORDS)
    if "consign" in text or has_keyword:
        return True
    return False


def filter_topic(posts: List[Dict]) -> List[Dict]:
    return [post for post in posts if matches_topic(post)]


def bucket_keywords(posts: List[Dict], *, limit: int = 5) -> List[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for post in posts:
        tokens = re.findall(r"[A-Za-z']+", post["clean_text"].lower())
        for token in tokens:
            if len(token) <= 3 or token in STOP_WORDS:
                continue
            counter[token] += 1
    return counter.most_common(limit)


def summarize(posts: List[Dict]) -> Dict:
    if not posts:
        return {
            "count": 0,
            "avg_sentiment": 0.0,
            "label_counts": {},
            "top_positive": [],
            "top_negative": [],
            "keywords": [],
            "negative_keywords": [],
        }
    avg_sent = mean(p["sentiment"] for p in posts)
    label_counts = Counter(p["sentiment_label"] for p in posts)
    sorted_desc = sorted(posts, key=lambda p: p["sentiment"], reverse=True)
    sorted_asc = sorted(posts, key=lambda p: p["sentiment"])
    top_positive = sorted_desc[:2]
    top_negative = sorted_asc[:2]
    keywords = bucket_keywords(posts)
    negative_keywords = bucket_keywords([p for p in posts if p["sentiment_label"] == "negative"], limit=5)
    return {
        "count": len(posts),
        "avg_sentiment": round(avg_sent, 3),
        "label_counts": label_counts,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "keywords": keywords,
        "negative_keywords": negative_keywords,
    }


def format_post(post: Dict) -> str:
    snippet = post["clean_text"]
    if len(snippet) > 240:
        snippet = snippet[:237].rstrip() + "..."
    return (
        f"- {post.get('title') or post.get('author')} (score {post['sentiment']:+.2f}): "
        f"{snippet}"
    )


def build_markdown(reddit_stats: Dict) -> str:
    lines = [
        "# Poshmark Consignment Social Sentiment",
        "",
        "## Volume snapshot",
        "Platform | Posts | Avg compound | + | = | -",
        ":-- | --: | --: | --: | --: | --:",
        f"Reddit | {reddit_stats['count']} | {reddit_stats['avg_sentiment']:+.2f} | "
        f"{reddit_stats['label_counts'].get('positive', 0)} | "
        f"{reddit_stats['label_counts'].get('neutral', 0)} | "
        f"{reddit_stats['label_counts'].get('negative', 0)}",
        "",
        "## Reddit highlights",
    ]
    def unique_posts(groups: List[List[Dict]]) -> List[Dict]:
        seen = set()
        ordered: List[Dict] = []
        for group in groups:
            for post in group:
                key = post.get("url") or post.get("title") or post.get("author")
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(post)
        return ordered

    for post in unique_posts([reddit_stats["top_positive"], reddit_stats["top_negative"]]):
        lines.append(format_post(post))
    lines.append("")
    if reddit_stats["keywords"]:
        keywords = ", ".join(f"{word} ({count})" for word, count in reddit_stats["keywords"])
        lines.append(f"Top recurring terms: {keywords}")
    if reddit_stats["negative_keywords"]:
        keywords = ", ".join(f"{word} ({count})" for word, count in reddit_stats["negative_keywords"])
        lines.append(f"Pain-point terms: {keywords}")

    lines.append("")
    lines.append("## Methodology & caveats")
    lines.append("- Reddit snippets sourced from Brave Search infoboxes due to direct API restrictions.")
    lines.append(f"- Limited to posts retrieved within the past {WINDOW_DAYS} days that mention consignors sending bags to partners.")
    lines.append("- Where Reddit timestamps were unavailable, retrieval time (UTC) was used as a proxy.")
    lines.append("- Compound score thresholds: ≥0.05 positive, ≤-0.05 negative.")
    return "\n".join(lines)


def main() -> None:
    reddit_raw = add_sentiment(load_json(DATA_DIR / "reddit_posts.json"))
    reddit_filtered = filter_topic(filter_recent(reddit_raw))
    reddit_stats = summarize(reddit_filtered)
    output = {
        "reddit": reddit_stats,
    }
    (REPORT_DIR / "poshmark_sentiment.json").write_text(json.dumps(output, indent=2))
    markdown = build_markdown(reddit_stats)
    (REPORT_DIR / "poshmark_sentiment.md").write_text(markdown)
    print("Saved:")
    print(" -", REPORT_DIR / "poshmark_sentiment.json")
    print(" -", REPORT_DIR / "poshmark_sentiment.md")


if __name__ == "__main__":
    main()
