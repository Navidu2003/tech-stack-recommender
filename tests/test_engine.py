"""
Tech Stack Recommender — Unit Tests
Run: python tests/test_engine.py
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.engine import TFIDFVectorizer, SimilarityEngine, RecommendationEngine


# ─────────────────────────────────────────────
# TFIDFVectorizer Tests
# ─────────────────────────────────────────────

class TestTFIDFVectorizer(unittest.TestCase):

    def setUp(self):
        self.vectorizer = TFIDFVectorizer()
        self.docs = [
            "python sql machine_learning",
            "javascript react html css",
            "docker kubernetes linux devops",
            "python tensorflow deep_learning",
        ]

    def test_fit_builds_vocabulary(self):
        self.vectorizer.fit(self.docs)
        self.assertGreater(len(self.vectorizer.vocabulary), 0)

    def test_vocabulary_contains_all_terms(self):
        self.vectorizer.fit(self.docs)
        self.assertIn("python", self.vectorizer.vocabulary)
        self.assertIn("javascript", self.vectorizer.vocabulary)
        self.assertIn("docker", self.vectorizer.vocabulary)

    def test_fit_computes_idf_values(self):
        self.vectorizer.fit(self.docs)
        self.assertIn("python", self.vectorizer.idf_values)
        self.assertGreater(self.vectorizer.idf_values["python"], 0)

    def test_transform_returns_correct_vector_count(self):
        self.vectorizer.fit(self.docs)
        vectors = self.vectorizer.transform(self.docs)
        self.assertEqual(len(vectors), len(self.docs))

    def test_transform_returns_correct_vector_length(self):
        self.vectorizer.fit(self.docs)
        vectors = self.vectorizer.transform(self.docs)
        vocab_size = len(self.vectorizer.vocabulary)
        for v in vectors:
            self.assertEqual(len(v), vocab_size)

    def test_transform_before_fit_raises_error(self):
        v = TFIDFVectorizer()
        with self.assertRaises(RuntimeError):
            v.transform(["python sql"])

    def test_fit_transform_equals_fit_then_transform(self):
        v1 = TFIDFVectorizer()
        result_fit_transform = v1.fit_transform(self.docs)

        v2 = TFIDFVectorizer()
        v2.fit(self.docs)
        result_separate = v2.transform(self.docs)

        self.assertEqual(result_fit_transform, result_separate)

    def test_unknown_term_gets_zero_weight(self):
        self.vectorizer.fit(self.docs)
        vector = self.vectorizer.transform(["unknownxyz_skill_abc"])[0]
        self.assertEqual(sum(vector), 0.0)

    def test_repeated_term_has_higher_tf_than_single_occurrence(self):
        """A document where python appears 3/3 terms should have TF=1.0,
        while a document where python appears 1/3 terms should have TF=0.33."""
        self.vectorizer.fit(["python python python", "sql java python"])
        # "python python python" -> TF for python = 3/3 = 1.0
        # "sql java python"      -> TF for python = 1/3 = 0.33
        v1 = self.vectorizer.transform(["python python python"])[0]
        v2 = self.vectorizer.transform(["sql java python"])[0]
        idx = self.vectorizer.vocabulary.get("python", -1)
        if idx >= 0:
            self.assertGreater(v1[idx], v2[idx])


# ─────────────────────────────────────────────
# SimilarityEngine Tests
# ─────────────────────────────────────────────

class TestSimilarityEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SimilarityEngine()

    def test_identical_vectors_score_one(self):
        v = [0.5, 0.3, 0.8, 0.1]
        score = self.engine.cosine_similarity(v, v)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_opposite_vectors_score_zero(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        score = self.engine.cosine_similarity(v1, v2)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_zero_vector_returns_zero(self):
        score = self.engine.cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        self.assertEqual(score, 0.0)

    def test_score_is_between_zero_and_one(self):
        v1 = [0.2, 0.5, 0.8, 0.1]
        v2 = [0.1, 0.4, 0.9, 0.3]
        score = self.engine.cosine_similarity(v1, v2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_mismatched_vector_lengths_raise_error(self):
        with self.assertRaises(ValueError):
            self.engine.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_rank_similarities_returns_sorted_results(self):
        query = [1.0, 0.0, 0.0]
        corpus = [
            [0.5, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        ranked = self.engine.rank_similarities(query, corpus)
        scores = [score for _, score in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_similarities_returns_all_indices(self):
        query = [1.0, 0.0]
        corpus = [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]
        ranked = self.engine.rank_similarities(query, corpus)
        indices = sorted([idx for idx, _ in ranked])
        self.assertEqual(indices, [0, 1, 2])


# ─────────────────────────────────────────────
# RecommendationEngine Tests
# ─────────────────────────────────────────────

class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RecommendationEngine()
        self.engine.load_data()

    def test_data_loads_successfully(self):
        self.assertTrue(self.engine._loaded)
        self.assertGreater(len(self.engine.job_roles), 0)

    def test_dataset_has_expected_roles(self):
        roles = self.engine.get_all_roles()
        self.assertIn("Data Scientist", roles)
        self.assertIn("Full Stack Developer", roles)

    def test_recommend_returns_correct_count(self):
        results = self.engine.recommend(["python", "sql"], top_n=3)
        self.assertEqual(len(results), 3)

    def test_recommend_results_are_ranked(self):
        results = self.engine.recommend(["python", "machine_learning", "sql"], top_n=5)
        scores = [r.match_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_recommend_python_ml_skills(self):
        results = self.engine.recommend(["python", "machine_learning", "statistics"], top_n=1)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].match_score, 0)

    def test_recommend_frontend_skills(self):
        results = self.engine.recommend(["javascript", "react", "html", "css"], top_n=3)
        titles = [r.job_title for r in results]
        self.assertTrue(any("Frontend" in t or "Full Stack" in t for t in titles))

    def test_recommend_empty_skills_raises_error(self):
        with self.assertRaises(ValueError):
            self.engine.recommend([])

    def test_results_have_matched_skills_field(self):
        results = self.engine.recommend(["python", "sql"], top_n=3)
        for r in results:
            self.assertIsInstance(r.matched_skills, list)

    def test_results_have_missing_skills_field(self):
        results = self.engine.recommend(["python"], top_n=3)
        for r in results:
            self.assertIsInstance(r.missing_skills, list)

    def test_get_all_skills_returns_list(self):
        skills = self.engine.get_all_skills()
        self.assertIsInstance(skills, list)
        self.assertGreater(len(skills), 0)

    def test_get_all_roles_returns_list(self):
        roles = self.engine.get_all_roles()
        self.assertIsInstance(roles, list)
        self.assertGreater(len(roles), 0)


# ─────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTFIDFVectorizer))
    suite.addTests(loader.loadTestsFromTestCase(TestSimilarityEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendationEngine))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"\n{'=' * 50}")
    print(f"  {passed}/{total} tests passed")
    if result.wasSuccessful():
        print("  All tests passed!")
    print(f"{'=' * 50}\n")

    sys.exit(0 if result.wasSuccessful() else 1)
