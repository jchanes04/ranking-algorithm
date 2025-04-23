from __future__ import annotations
from typing import Iterable, List, Set, TypedDict
from numpy.typing import NDArray
import numpy as np

class _MatchDataRow(TypedDict):
    team1Name: str
    team2Name: str
    team1Score: float
    team2Score: float

class MatchScores():
    def __init__(self, scores: NDArray, team_names):
        self._scores = scores
        self._team_names = team_names

    def team_names_from_dataframe(data_rows: Iterable[_MatchDataRow]):
        team_names_set: Set[str] = set()
        for row in data_rows:
            team_names_set.add(row["team1Name"])
            team_names_set.add(row["team2Name"])

        return list(team_names_set)

    def from_df(df, team_names=None):
        team_numbers = dict()
        data_rows = [row for _, row in df.iterrows()]
        if not team_names:
            team_names = MatchScores.team_names_from_dataframe(data_rows)

        for i, name in enumerate(team_names):
            team_numbers[name] = i

        scores = np.array([
            [
                team_numbers[x["team1Name"]],
                team_numbers[x["team2Name"]],
                x["team1Score"],
                x["team2Score"]
            ] for x in data_rows
        ])

        return MatchScores(scores, team_names)
    
    def pick_matches(self, indices: List[int]) -> MatchScores:
        picked_rows = np.array([
            x for i, x in enumerate(self.scores)
            if i in indices
        ])
        return MatchScores(picked_rows, self.team_names)
    
    @property
    def scores(self):
        return self._scores
    
    @property
    def team_names(self):
        return self._team_names