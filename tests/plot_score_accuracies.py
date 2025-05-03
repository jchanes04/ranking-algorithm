# tests/test.py
import sys
from pathlib import Path

# Add the `src/` directory to Python's search path
root_dir = Path(__file__).parent.parent  # Gets the project root (parent of "tests/")
sys.path.append(str(root_dir / "src"))  # Now Python can see "src/ranking_algorithm"

from ranking_algorithm import RatingEngine
from plots import score_accuracy_scatter
import pandas as pd
import re

group_pattern = r"([a-zA-Z]+)[0-9]*.csv"

datasets = [
    "hockey2000.csv",
    "hockey2001.csv",
    "hockey2002.csv",
    "hockey2005.csv",
    "hockey2006.csv",
    "hockey2007.csv",
    "hockey2008.csv",
    "hockey2010.csv",
    "hockey2014.csv",
    "hockey2015.csv",
    "hockey2016.csv",
    "hockey2017.csv",
    "hockey2019.csv",
    "baseball2010.csv",
    "baseball2011.csv",
    "baseball2012.csv",
    "baseball2014.csv",
    "baseball2015.csv",
    "baseball2016.csv",
    "baseball2018.csv",
    "baseball2019.csv",
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
    "scibowl4.csv",
    "scibowl5.csv",
    "scibowl6.csv",
    "scibowl7.csv",
    "rocketLeague.csv"
]
num_games = {}
score_accuracies = {}
for dataset in datasets:
    print(f"Analyzing {dataset}...")

    group_match = re.match(group_pattern, dataset)
    group_name = group_match.group(1)

    if group_name not in num_games:
        num_games[group_name] = []
    if group_name not in score_accuracies:
        score_accuracies[group_name] = []

    df = pd.read_csv(f"tests/data/{dataset}")
    num_games[group_name].append(len(df))
    engine = RatingEngine.from_dataframe(df)

    score_accuracy = engine.score_accuracy()
    score_accuracies[group_name].append((score_accuracy, dataset))

fig = score_accuracy_scatter(num_games, score_accuracies)
fig.update_layout(title_text="Score Accuracies")
fig.update_xaxes(title_text="Number of Matches")
fig.update_yaxes(title_text="Accuracy Rate")
# fig.show()
print("writing...")
fig.write_image("tests/plots/score_accuracies.png", scale=3)
print("wrote!")