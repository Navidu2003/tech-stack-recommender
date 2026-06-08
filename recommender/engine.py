"""
Tech Stack Recommender - Core AI Engine
Uses TF-IDF Vectorization + Cosine Similarity (built from scratch, no external ML libraries)
"""

import csv
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class JobRole:
    """Represents a job role with its required tech skills."""
    title: str
    skills: List[str]
    category: str
    description: str


@dataclass
class RecommendationResult:
    """A single recommendation result."""
    rank: int
    job_title: str
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    category: str
    description: str


class TFIDFVectorizer:
    """
    TF-IDF Vectorizer built from scratch.
    
    TF  (Term Frequency)     = how often a skill appears in a role
    IDF (Inverse Doc Freq)   = how rare/important a skill is across all roles
    TF-IDF                   = TF * IDF (higher = more important/relevant)
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf_values: Dict[str, float] = {}
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Convert text to lowercase skill tokens."""
        return [token.strip().lower() for token in text.replace(",", " ").split() if token.strip()]

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Term Frequency: count of each term / total terms."""
        tf = {}
        total = len(tokens) if tokens else 1
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        return {term: count / total for term, count in tf.items()}

    def _compute_idf(self, documents: List[List[str]]) -> None:
        """Inverse Document Frequency: log(N / df) for each term."""
        n_docs = len(documents)
        df = {}
        for doc in documents:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.idf_values = {
            term: math.log((n_docs + 1) / (count + 1)) + 1
            for term, count in df.items()
        }

    def fit(self, documents: List[str]) -> "TFIDFVectorizer":
        """Learn vocabulary and IDF values from a list of documents."""
        tokenized = [self._tokenize(doc) for doc in documents]
        all_terms = set(term for doc in tokenized for term in doc)
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
        self._compute_idf(tokenized)
        self.fitted = True
        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """Convert documents to TF-IDF vectors."""
        if not self.fitted:
            raise RuntimeError("Vectorizer must be fitted before transform.")
        vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tf = self._compute_tf(tokens)
            vector = [0.0] * len(self.vocabulary)
            for term, idx in self.vocabulary.items():
                if term in tf:
                    vector[idx] = tf[term] * self.idf_values.get(term, 1.0)
            vectors.append(vector)
        return vectors

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


class SimilarityEngine:
    """Computes Cosine Similarity between vectors."""

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Cosine Similarity = dot_product(A, B) / (|A| * |B|)
        Returns a value between 0.0 (no match) and 1.0 (perfect match).
        """
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be the same length.")

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
        magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def rank_similarities(
        self, query_vector: List[float], corpus_vectors: List[List[float]]
    ) -> List[Tuple[int, float]]:
        """Return (index, score) pairs sorted by similarity descending."""
        scores = [
            (idx, self.cosine_similarity(query_vector, vec))
            for idx, vec in enumerate(corpus_vectors)
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)


class RecommendationEngine:
    """
    Main recommendation engine.
    Loads job roles, vectorizes skills, and recommends roles based on user input.
    """

    def __init__(self, data_path: str = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "raw_skills.csv")

        self.data_path = data_path
        self.job_roles: List[JobRole] = []
        self.vectorizer = TFIDFVectorizer()
        self.similarity_engine = SimilarityEngine()
        self.role_vectors: List[List[float]] = []
        self._loaded = False

    def load_data(self) -> None:
        """Load job roles from CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        with open(self.data_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                skills = [s.strip().lower() for s in row["skills"].split(",") if s.strip()]
                self.job_roles.append(JobRole(
                    title=row["title"].strip(),
                    skills=skills,
                    category=row.get("category", "General").strip(),
                    description=row.get("description", "").strip(),
                ))

        skill_documents = [" ".join(role.skills) for role in self.job_roles]
        self.role_vectors = self.vectorizer.fit_transform(skill_documents)
        self._loaded = True

    def recommend(self, user_skills: List[str], top_n: int = 5) -> List[RecommendationResult]:
        """
        Recommend top N job roles based on user skills.
        
        Args:
            user_skills: List of skills the user has
            top_n: Number of recommendations to return
        
        Returns:
            List of RecommendationResult objects ranked by match score
        """
        if not self._loaded:
            self.load_data()

        if not user_skills:
            raise ValueError("Please provide at least one skill.")

        user_doc = " ".join([s.strip().lower() for s in user_skills])
        user_vector = self.vectorizer.transform([user_doc])[0]
        ranked = self.similarity_engine.rank_similarities(user_vector, self.role_vectors)

        results = []
        user_skill_set = {s.strip().lower() for s in user_skills}

        for rank, (idx, score) in enumerate(ranked[:top_n], start=1):
            role = self.job_roles[idx]
            role_skill_set = set(role.skills)
            matched = sorted(user_skill_set & role_skill_set)
            missing = sorted(role_skill_set - user_skill_set)[:5]

            results.append(RecommendationResult(
                rank=rank,
                job_title=role.title,
                match_score=round(score * 100, 2),
                matched_skills=matched,
                missing_skills=missing,
                category=role.category,
                description=role.description,
            ))

        return results

    def get_all_roles(self) -> List[str]:
        """Return list of all available job role titles."""
        if not self._loaded:
            self.load_data()
        return [role.title for role in self.job_roles]

    def get_all_skills(self) -> List[str]:
        """Return sorted list of all unique skills in the dataset."""
        if not self._loaded:
            self.load_data()
        all_skills = set()
        for role in self.job_roles:
            all_skills.update(role.skills)
        return sorted(all_skills)
