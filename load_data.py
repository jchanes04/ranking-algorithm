import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Iterable, TypedDict
from numpy.typing import NDArray

class MatchDataRow(TypedDict):
    team1Name: str
    team2Name: str
    team1Score: float
    team2Score: float
    

def generate_match_scores(data_rows: Iterable[MatchDataRow]) -> Tuple[NDArray[np.float64], List[str]]:
    team_names_set: Set[str] = set()
    for row in data_rows:
        team_names_set.add(row["team1Name"])
        team_names_set.add(row["team2Name"])

    team_names = list(team_names_set)
    team_numbers = dict()
    for i, name in enumerate(team_names):
        team_numbers[name] = i

    match_scores = np.array([
        [team_numbers[x["team1Name"]], team_numbers[x["team2Name"]], x["team1Score"], x["team2Score"]] for x in data_rows
    ])
    return (match_scores, team_names)

def get_num_games(match_scores: NDArray, team_names: List[str]):
    result = {}
    for team in team_names:
        result[team] = 0
    for m in match_scores:
        team_1_number, team_2_number, _, _ = m
        result[team_names[team_1_number]] += 1
        result[team_names[team_2_number]] += 1
    return result

def coeff_rows_from_match(num_games: Dict[str, int], team_names: List[str], match_row: Tuple[int, int, int, int]) -> List[NDArray[np.float64]]:
    num_teams = len(num_games.keys())
    team_1_number, team_2_number, team_1_score, team_2_score = match_row
    assert team_1_number != team_2_number

    team_1_name = team_names[team_1_number]
    team_2_name = team_names[team_2_number]

    total_score = team_1_score + team_2_score
    match_scalings = [1/np.sqrt(num_games[team_1_name]), 1/np.sqrt(num_games[team_2_name])]

    result = []
    for scal in match_scalings:
        if team_1_number < team_2_number:
            n1 = team_1_number
            n2 = team_2_number
            s1 = (team_1_score/total_score) * scal
            s2 = (team_2_score/total_score) * scal
        else:
            n1 = team_2_number
            n2 = team_1_number
            s1 = (team_2_score/total_score) * scal
            s2 = (team_1_score/total_score) * scal
        
        result.append(np.concatenate([
            np.zeros(n1),
            [s2],
            np.zeros(n2 - n1 - 1),
            [-s1],
            np.zeros(num_teams - n2 - 1)
        ]))
    return result

def generate_coeff_matrix(team_names: List[str], match_scores: NDArray) -> NDArray:
    num_games = get_num_games(match_scores, team_names)
    match_rows = [
        coeff_rows_from_match(num_games, team_names, m) for m in match_scores
    ]

    all_rows: List[NDArray] = []
    for row_list in match_rows:
        all_rows.extend(row_list)
    return np.array([ *all_rows ])