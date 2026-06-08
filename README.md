# Tech Stack Recommender 🤖

An AI-powered job role recommendation system built **entirely from scratch** — no scikit-learn, no external ML libraries. Uses **TF-IDF Vectorization** and **Cosine Similarity** implemented in pure Python.

> Built during DecodeLabs Industrial Training, Batch 2026

---

## What It Does

Enter your skills → get ranked job role recommendations with match scores.

```
Input:  python, sql, machine_learning, statistics

Output:
  #1  Data Scientist       Match: 96%  ██████████
  #2  ML Engineer          Match: 89%  ████████░░
  #3  Data Analyst         Match: 74%  ███████░░░
```

---

## How the AI Works

### TF-IDF (Term Frequency — Inverse Document Frequency)
Converts skill words into numbers that reflect their importance:
- **TF** = how often a skill appears in a job role's profile
- **IDF** = how rare/unique that skill is across all roles
- **TF-IDF** = TF × IDF → higher score = more important/distinctive skill

### Cosine Similarity
Measures the angle between two skill vectors (your profile vs each job role):
- **1.0** = perfect match (same direction)
- **0.0** = no overlap at all
- Formula: `cos(θ) = (A · B) / (|A| × |B|)`

All of this is coded from scratch in `recommender/engine.py` — no black boxes.

---

## Project Structure

```
tech-stack-recommender/
│
├── main.py                  ← Entry point (run this)
│
├── recommender/
│   ├── __init__.py          ← Package exports
│   └── engine.py            ← Core AI logic (TF-IDF + Cosine Similarity)
│
├── data/
│   └── raw_skills.csv       ← Dataset: 20 job roles with required skills
│
├── tests/
│   └── test_engine.py       ← 27 unit tests
│
├── requirements.txt         ← Zero external dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### Requirements
- Python 3.8 or higher
- No pip installs needed — zero external dependencies

### Run the project

**Interactive mode** (recommended for first time):
```bash
python main.py
```

**Demo mode** (see 3 sample profiles instantly):
```bash
python main.py --demo
```

**CLI mode** (pass skills directly):
```bash
python main.py --skills "python sql machine_learning"
```

**With custom number of results:**
```bash
python main.py --skills "react html css javascript" --top 5
```

**Run all tests:**
```bash
python tests/test_engine.py
```

**List all available skills:**
```bash
python main.py --list-skills
```

**List all available job roles:**
```bash
python main.py --list-roles
```

---

## Dataset

20 job roles across 6 categories:

| Category | Roles |
|---|---|
| Data & AI | Data Scientist, ML Engineer, Data Analyst, Data Engineer, AI Research Scientist, NLP Engineer, Computer Vision Engineer |
| Web Development | Full Stack Developer, Frontend Developer, Backend Developer |
| Infrastructure | DevOps Engineer, Cloud Architect, Site Reliability Engineer |
| Security | Cybersecurity Engineer |
| Mobile | Mobile Developer |
| Emerging Tech | Blockchain Developer |
| Creative Tech | Game Developer |
| Finance & Data | Quantitative Analyst |
| Design | UI/UX Designer |
| Hardware | Embedded Systems Engineer |

---

## Classes

| Class | Purpose |
|---|---|
| `TFIDFVectorizer` | Builds vocabulary, computes TF-IDF vectors from scratch |
| `SimilarityEngine` | Computes Cosine Similarity between vectors |
| `RecommendationEngine` | Orchestrates loading, vectorizing, and ranking |
| `JobRole` | Data model for a job role |
| `RecommendationResult` | Data model for a recommendation result |

---

## Tests

27 unit tests covering all three core classes:

```bash
python tests/test_engine.py
```

Expected output:
```
test_data_loads_successfully ... ok
test_dataset_has_expected_roles ... ok
test_fit_builds_vocabulary ... ok
...
27/27 tests passed ✓
```

---

## Extending the Dataset

Add your own job roles to `data/raw_skills.csv`:

```csv
title,category,skills,description
Prompt Engineer,Data & AI,"python,nlp,llm,api_development,prompt_design","Designs and optimises prompts for large language models."
```

The engine automatically picks up new rows — no code changes needed.

---

## Tech Stack

- **Language**: Python 3.8+
- **Algorithm**: TF-IDF + Cosine Similarity (pure Python)
- **Data**: CSV
- **Tests**: `unittest` (standard library)
- **External libraries**: None

---

## Author

Built by DecodeLabs Industrial Training Batch 2026  
[GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/)

---

## License

MIT License — free to use, modify, and distribute.
