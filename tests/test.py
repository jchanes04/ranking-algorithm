# tests/test.py
import sys
from pathlib import Path

# Add the `src/` directory to Python's search path
root_dir = Path(__file__).parent.parent  # Gets the project root (parent of "tests/")
sys.path.append(str(root_dir / "src"))  # Now Python can see "src/ranking_algorithm"

from ranking_algorithm import RatingEngine
from plots import accuracy_curves
import pandas as pd

datasets = [
    "basketball1980.csv",
    "basketball1984.csv",
    "basketball1985.csv",
    "basketball1987.csv",
    "basketball1992.csv",
    "basketball1993.csv",
    "basketball1995.csv",
    "basketball1999.csv",
    "basketball2003.csv",
    "basketball2008.csv",
    "basketball2011.csv",
    "basketball2012.csv",
    "basketball2016.csv",
    "basketball2017.csv",
    "basketball2019.csv",
    "basketball2024.csv",
    "football1999.csv",
    "football2000.csv",
    "football2001.csv",
    "football2002.csv",
    "football2008.csv",
    "football2009.csv",
    "football2010.csv",
    "football2011.csv",
    "football2013.csv",
    "football2015.csv",
    "football2017.csv",
    "football2018.csv",
    "football2024.csv",
    "scibowl1.csv",
    "scibowl2.csv",
    "scibowl3.csv",
    # "scibowl4.csv",
    "scibowl5.csv",
    "scibowl6.csv",
    "scibowl7.csv"
]
num_games = []
score_accuracies = []
win_accuracies = []
for dataset in datasets:
    print(f"Analyzing {dataset}...")
    df = pd.read_csv(f"data/{dataset}")
    num_games.append(len(df))
    engine = RatingEngine.from_dataframe(df)
    ratings = engine.compute_ratings()

    score_accuracy = engine.k_fold()
    score_accuracies.append((score_accuracy, dataset))

    win_accuracy = engine.win_accuracy(ratings)
    win_accuracies.append((win_accuracy, dataset))

fig = accuracy_curves(num_games, {
    "Score Accuracy": score_accuracies,
    "Win Accuracy": win_accuracies
})
fig.show()