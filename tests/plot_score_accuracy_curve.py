# tests/test.py
import sys
from pathlib import Path

# Add the `src/` directory to Python's search path
root_dir = Path(__file__).parent.parent  # Gets the project root (parent of "tests/")
sys.path.append(str(root_dir / "src"))  # Now Python can see "src/ranking_algorithm"

from ranking_algorithm import RatingEngine
from plots import score_accuracy_scatter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import optimize
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
score_accuracies = {}
for dataset in datasets:    
    print(f"Analyzing {dataset}...")

    group_match = re.match(group_pattern, dataset)
    group_name = group_match.group(1)

    if group_name not in score_accuracies:
        score_accuracies[group_name] = []

    df = pd.read_csv(f"tests/data/{dataset}")
    engine = RatingEngine.from_dataframe(df)

    for k in range(3, 15):
        for r in range(10, 12):
            print(f"({k}, {r})")
            score_accuracy = engine.score_accuracy(k=k, random_state=r)
            score_accuracies[group_name].append(((k-1)/k, score_accuracy, dataset))

avg_points = {}
for group_name, data in score_accuracies.items():
    grouped_data = {}
    avg_points[group_name] = []

    for frac, acc, _ in data:
        if frac not in grouped_data:
            grouped_data[frac] = []
        grouped_data[frac].append(acc)
    
    for frac, acc_list in grouped_data.items():
        avg = np.mean(acc_list)
        std = np.std(acc_list)
        avg_points[group_name].append((frac, avg, std))

def fit_func(x, a, b, c):
    y = a*x**2 + b*x + c
    return y
    

colors=px.colors.qualitative.Plotly

fig = go.Figure()
for i, (group_name, points) in enumerate(avg_points.items()):
    xdata, ydata, yerr = zip(*points)
    popt, pcov = optimize.curve_fit(fit_func, xdata=list(xdata), ydata=list(ydata), sigma=list(yerr))
    ypred = fit_func(np.array(xdata), popt[0], popt[1], popt[2])

    error_bar_settings = {
        "type": "data",
        "array": yerr,
        "visible": True
    }

    scatter_trace = go.Scatter(
        x=list(xdata),
        y=list(ydata),
        error_y=error_bar_settings,
        mode="markers",
        name=f"{group_name} Data",
        marker_color=colors[i]
    )
    fit_trace = go.Scatter(
        x=xdata,
        y=ypred,
        mode="lines",
        name=f"{group_name} Fit",
        marker_color=colors[i]
    )
    fig.add_trace(scatter_trace)
    fig.add_trace(fit_trace)
fig.update_xaxes(type="log")
fig.update_layout(title_text="Score Accuracy Curve")
fig.update_xaxes(title_text="Fraction of Training Data")
fig.update_yaxes(title_text="Accuracy Rate")
fig.show()

print("writing...")
fig.write_image("tests/plots/score_accuracy_curve.png", scale=3)
print("wrote!")