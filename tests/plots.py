import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ranking_algorithm import Ratings
from typing import Dict, List, Tuple

def rating_hist(ratings: Ratings, nbins=8):
    fig = px.histogram(ratings.rating_dict.values(), nbins=nbins)
    return fig

def accuracy_curves(num_games: List[int], curves: Dict[str, List[Tuple[float, str]]]):
    fig = go.Figure()
    for name, data in curves.items():
        acc_list, labels = zip(*data)
        trace = go.Scatter(x=num_games, y=acc_list, text=labels, name=name, mode="markers")
        fig.add_trace(trace)
    fig.update_xaxes(type="log")
    return fig

def win_accuracy_scatter(num_games: Dict[str, List[int]], win_accs: Dict[str, Tuple[float, str]]):
    fig = go.Figure()
    for name, data in win_accs.items():
        acc_list, labels = zip(*data)
        trace = go.Scatter(x=num_games[name], y=acc_list, text=labels, name=name, mode="markers")
        fig.add_trace(trace)
    fig.update_xaxes(type="log")
    return fig

def score_accuracy_scatter(num_games: Dict[str, List[int]], win_accs: Dict[str, Tuple[float, str]]):
    fig = go.Figure()
    for name, data in win_accs.items():
        acc_list, labels = zip(*data)
        trace = go.Scatter(x=num_games[name], y=acc_list, text=labels, name=name, mode="markers")
        fig.add_trace(trace)
    fig.update_xaxes(type="log")
    return fig

def win_accuracy_curves(num_games: Dict[str, List[int]], win_accs: Dict[str, Tuple[float, str]]):
    fig = go.Figure()
    for name, data in win_accs.items():
        acc_list, labels = zip(*data)
        trace = go.Scatter(x=num_games[name], y=acc_list, text=labels, name=name, mode="markers")
        fig.add_trace(trace)
    fig.update_xaxes(type="log")
    return fig