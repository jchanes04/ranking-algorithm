import numpy as np
import scipy
from load_data import generate_coeff_matrix
from typing import Dict, List, Tuple

def compute_ratings(match_scores, team_names, avg=100):
    num_teams = len(team_names)
    C = generate_coeff_matrix(team_names, match_scores)

    CTC = np.dot(C.T, C)
    constrained = np.block([[CTC, np.ones((num_teams, 1))], [np.ones((1, num_teams)), 0]])
    rhs = np.vstack((np.zeros((num_teams, 1)), 1))

    coeffs = scipy.linalg.solve(constrained, rhs)
    result: Dict[str, float] = {}
    for i, [coeff] in enumerate(list(coeffs[:-1])):
        result[team_names[i]] = coeff * avg * num_teams
    return result