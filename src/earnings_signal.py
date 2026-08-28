"""
earnings_signal.py

Scores earnings-disclosure text for directional sentiment, producing a
continuous score analogous to the "continuous score" evaluated against
market reactions in my MSc dissertation:

    Semar, M. (2026). "Reading Between the Lines: Can an LLM Predict
    How Markets React to Earnings?" Citibank Applied Research Project,
    Bayes Business School.

That dissertation found the model's continuous score correlated with
overnight returns (Spearman rho = 0.2565, p = 0.0001, n = 109), and
benchmarked the model against a Loughran-McDonald lexicon baseline
among others. This module implements that lexicon-based baseline as a
free, dependency-light, fully local scorer — designed as a clean
drop-in replacement point for a real LLM call later.

=== Upgrade path ===
This module intentionally exposes a single interface,
`EarningsSignalScorer.score(text) -> float`, so a production version
can swap in a real LLM (e.g. the Anthropic API) without touching any
downstream code in earnings_var.py. See `LLMSignalScorer` below for
the shape that upgrade would take (not implemented/called here).

=== Lexicon note ===
LEXICON below is a small, hand-curated, illustrative finance-sentiment
word list — NOT the full Loughran-McDonald Master Dictionary (which is
free for academic/research use at
https://sraf.nd.edu/loughranmcdonald-master-dictionary/ but requires a
separate download/licence check before redistribution). Swap in the
full LM dictionary for anything beyond a demo/prototype.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


# Small illustrative finance-sentiment lexicon, in the same category
# structure as the real Loughran-McDonald dictionary (positive,
# negative, uncertainty, litigious). Extend or replace with the full
# LM Master Dictionary for production use.
LEXICON: dict[str, set[str]] = {
    "positive": {
        "growth", "grew", "growing", "record", "strong", "strength", "beat", "beats",
        "exceeded", "exceeding", "outperform", "outperformed", "improve", "improved",
        "improvement", "robust", "accelerate", "accelerated", "acceleration", "gain",
        "gains", "profitable", "profitability", "margin", "expansion", "expand",
        "expanded", "upgrade", "upgraded", "raise", "raised", "raising",
        "momentum", "resilient", "resilience", "surge", "surged", "rebound", "recovery",
        "optimistic", "optimism", "opportunity", "opportunities", "innovation",
        "innovative", "leadership", "milestone", "successful", "success", "solid",
        "healthy", "confident", "confidence", "diversified", "efficiency", "efficient",
    },
    "negative": {
        "decline", "declined", "declining", "decrease", "decreased", "decreasing",
        "weak", "weakness", "weaker", "miss", "missed", "missing", "shortfall",
        "below", "underperform", "underperformed", "loss", "losses", "impairment",
        "writedown", "write-down", "headwind", "headwinds", "challenge", "challenging",
        "challenges", "pressure", "pressured", "slowdown", "slowing", "contraction",
        "contracted", "layoff", "layoffs", "restructuring", "downgrade", "downgraded",
        "lower", "lowered", "lowering", "cut", "cuts", "reduced", "reduction",
        "disruption", "disrupted", "volatile", "volatility", "uncertainty", "risk",
        "risks", "concern", "concerns", "cautious", "caution", "delay", "delayed",
        "default", "bankruptcy", "recession", "deficit", "warn", "warning", "warned",
    },
    "uncertainty": {
        "may", "might", "could", "uncertain", "uncertainty", "unpredictable",
        "unclear", "possibly", "approximately", "estimate", "estimated", "estimates",
        "assume", "assumption", "contingent", "depends", "variability", "fluctuate",
        "fluctuation", "pending", "tbd", "preliminary",
    },
    "litigious": {
        "lawsuit", "litigation", "settlement", "regulatory", "investigation",
        "compliance", "penalty", "fine", "sanctions", "subpoena", "allegation",
        "allegations", "breach", "violation",
    },
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s\-]", " ", text)
    return text.split()


@dataclass
class EarningsSignalScorer:
    """
    Lexicon-based earnings disclosure scorer.

    score(text) returns a continuous value roughly in [-1, 1]:
    positive = bullish tone, negative = bearish tone, magnitude
    reflects how lopsided positive/negative word counts are relative
    to total document length — deliberately analogous in spirit (not
    identical in mechanism) to the continuous score evaluated in the
    Citibank dissertation.
    """
    lexicon: dict[str, set[str]] = field(default_factory=lambda: LEXICON)

    def score(self, text: str) -> float:
        tokens = _tokenize(text)
        if not tokens:
            return 0.0

        n_words = len(tokens)
        n_pos = sum(1 for t in tokens if t in self.lexicon["positive"])
        n_neg = sum(1 for t in tokens if t in self.lexicon["negative"])

        # Net tone, scaled by document length — matches the standard
        # Loughran-McDonald "net tone" formula: (pos - neg) / total_words,
        # then lightly rescaled so typical disclosures land well inside [-1, 1].
        raw_score = (n_pos - n_neg) / n_words
        scaled_score = max(-1.0, min(1.0, raw_score * 15))  # empirical scaling factor
        return scaled_score

    def score_breakdown(self, text: str) -> dict:
        """Return the full category breakdown, useful for evidence/debugging
        (mirrors the Citibank dissertation's 'surfaces supporting evidence'
        requirement at Trust Framework Level 1 — see README)."""
        tokens = _tokenize(text)
        n_words = len(tokens) or 1
        counts = {
            category: sum(1 for t in tokens if t in words)
            for category, words in self.lexicon.items()
        }
        return {
            "n_words": n_words,
            "counts": counts,
            "score": self.score(text),
        }


# ---------------------------------------------------------------------------
# Upgrade path (not implemented/called): swap EarningsSignalScorer for this
# class to move from the free lexicon baseline to a real LLM call, following
# the exact model pipeline described in the Citibank dissertation (Section 3.3):
# structured score + supporting evidence + BUY/HOLD/SELL decision, temperature=0.
# ---------------------------------------------------------------------------
class LLMSignalScorer:
    """
    NOT IMPLEMENTED — reference shape for a future upgrade.

    A production version would call an LLM (e.g. the Anthropic API) with
    a prompt instructing it to read the disclosure and return a
    structured score + evidence + decision, exactly as in the Citibank
    dissertation's model pipeline. Implementing this requires an API
    key and is left as the natural next step once the lexicon baseline
    is validated end-to-end (see notebooks/08_earnings_event_overlay.ipynb).
    """
    def score(self, text: str) -> float:
        raise NotImplementedError(
            "Plug in a real LLM call here (e.g. api.anthropic.com/v1/messages) "
            "once the lexicon-based baseline has been validated."
        )
