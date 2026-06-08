"""
Tech Stack Recommender — Main Entry Point
=========================================
Run modes:
  python main.py                                          # Interactive mode
  python main.py --demo                                   # Demo mode (3 sample profiles)
  python main.py --skills "python sql machine_learning"  # CLI mode
  python main.py --skills "react html css" --top 5       # CLI with custom top N
"""

import sys
import os
import argparse

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import RecommendationEngine

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         TECH STACK RECOMMENDER  v1.0.0                  ║
║         AI-Powered Job Role Matching Engine             ║
║         TF-IDF + Cosine Similarity (from scratch)       ║
╚══════════════════════════════════════════════════════════╝
"""

DEMO_PROFILES = [
    {
        "name": "Backend Developer Profile",
        "skills": ["python", "sql", "docker", "kubernetes", "rest_api", "git", "linux"],
    },
    {
        "name": "Data Science Profile",
        "skills": ["python", "machine_learning", "statistics", "pandas", "numpy", "sql", "tensorflow"],
    },
    {
        "name": "Frontend Developer Profile",
        "skills": ["javascript", "react", "html", "css", "typescript", "git", "figma"],
    },
]


def print_separator(char="─", width=60):
    print(char * width)


def display_results(results, title="Recommendations"):
    """Pretty-print recommendation results."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    for result in results:
        print(f"\n  #{result.rank}  {result.job_title}")
        print(f"       Category    : {result.category}")
        print(f"       Match Score : {result.match_score}%  {'█' * int(result.match_score / 10)}{'░' * (10 - int(result.match_score / 10))}")
        
        if result.matched_skills:
            print(f"       Matched     : {', '.join(result.matched_skills)}")
        else:
            print(f"       Matched     : (none directly matched)")
        
        if result.missing_skills:
            print(f"       To learn    : {', '.join(result.missing_skills[:3])}{'...' if len(result.missing_skills) > 3 else ''}")
        
        if result.description:
            print(f"       About       : {result.description}")
        
        print_separator()


def interactive_mode(engine):
    """Interactive mode — asks the user for skills step by step."""
    print(BANNER)
    print("  Welcome to the Tech Stack Recommender!")
    print("  Enter your skills and we'll match you to the best job roles.\n")

    print("  Available skills (sample):")
    all_skills = engine.get_all_skills()
    sample = all_skills[:30]
    for i in range(0, len(sample), 6):
        print("  " + "  ".join(f"{s:<22}" for s in sample[i:i+6]))
    print(f"  ... and {len(all_skills) - 30} more\n")

    print_separator()
    print("  Enter your skills separated by commas or spaces.")
    print('  Example: python, sql, machine_learning, docker')
    print_separator()

    raw = input("\n  Your skills > ").strip()
    if not raw:
        print("\n  No skills entered. Exiting.")
        return

    skills = [s.strip().lower() for s in raw.replace(",", " ").split() if s.strip()]

    raw_top = input(f"\n  How many recommendations? (default: 5) > ").strip()
    top_n = int(raw_top) if raw_top.isdigit() and int(raw_top) > 0 else 5

    print(f"\n  Analysing {len(skills)} skill(s)...")
    results = engine.recommend(skills, top_n=top_n)
    display_results(results, title=f"Top {top_n} Matches for: {', '.join(skills)}")

    print("\n  Run again?  python main.py")
    print("  CLI mode?   python main.py --skills \"your skills here\" --top 3\n")


def demo_mode(engine):
    """Demo mode — shows 3 predefined profiles."""
    print(BANNER)
    print("  DEMO MODE — Showing 3 sample profiles\n")

    for profile in DEMO_PROFILES:
        print(f"\n  Profile: {profile['name']}")
        print(f"  Skills : {', '.join(profile['skills'])}")
        results = engine.recommend(profile["skills"], top_n=3)
        display_results(results, title=f"Top 3 Matches")

    print("\n  Try with your own skills:")
    print('  python main.py --skills "python docker kubernetes" --top 5\n')


def cli_mode(engine, skills_str, top_n):
    """CLI mode — skills passed as command-line argument."""
    skills = [s.strip().lower() for s in skills_str.replace(",", " ").split() if s.strip()]
    
    if not skills:
        print("  Error: No skills provided.")
        sys.exit(1)

    print(BANNER)
    print(f"  Skills  : {', '.join(skills)}")
    print(f"  Top N   : {top_n}\n")
    
    results = engine.recommend(skills, top_n=top_n)
    display_results(results, title=f"Top {top_n} Matches")


def main():
    parser = argparse.ArgumentParser(
        description="Tech Stack Recommender — AI-powered job role matching engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                       # Interactive mode
  python main.py --demo                                # Demo with 3 sample profiles
  python main.py --skills "python sql docker"          # Match your skills
  python main.py --skills "react html css" --top 5    # Custom number of results
        """
    )
    parser.add_argument("--skills", type=str, help="Space or comma-separated list of your skills")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations (default: 5)")
    parser.add_argument("--demo", action="store_true", help="Run demo mode with sample profiles")
    parser.add_argument("--list-skills", action="store_true", help="Show all available skills in the dataset")
    parser.add_argument("--list-roles", action="store_true", help="Show all available job roles in the dataset")

    args = parser.parse_args()

    engine = RecommendationEngine()

    try:
        engine.load_data()
    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
        print("  Make sure data/raw_skills.csv exists in the project folder.")
        sys.exit(1)

    if args.list_skills:
        print("\nAll available skills:")
        for skill in engine.get_all_skills():
            print(f"  {skill}")
        return

    if args.list_roles:
        print("\nAll available job roles:")
        for role in engine.get_all_roles():
            print(f"  {role}")
        return

    if args.demo:
        demo_mode(engine)
    elif args.skills:
        cli_mode(engine, args.skills, args.top)
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()
