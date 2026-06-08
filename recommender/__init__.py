"""
Tech Stack Recommender — AI Recommendation Package
Built from scratch using TF-IDF + Cosine Similarity (no external ML libraries)
"""

from .engine import RecommendationEngine, RecommendationResult, JobRole, TFIDFVectorizer, SimilarityEngine

__all__ = [
    "RecommendationEngine",
    "RecommendationResult",
    "JobRole",
    "TFIDFVectorizer",
    "SimilarityEngine",
]

__version__ = "1.0.0"
__author__ = "DecodeLabs Batch 2026"
