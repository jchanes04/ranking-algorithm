from __future__ import annotations
from typing import Dict, Iterable, List, Set, Tuple, TypedDict
from numpy.typing import NDArray
import numpy as np
import scipy
from sklearn.model_selection import KFold
from .Ratings import Ratings
from .MatchScores import MatchScores
from pandas import DataFrame

class RatingEngine():
    def __init__(self, match_scores: MatchScores, team_names: List[str]):
        self._match_scores = match_scores
        self._team_names = team_names

    @staticmethod
    def from_dataframe(df: DataFrame) -> RatingEngine:
        match_scores = MatchScores.from_df(df)
        return RatingEngine(match_scores, match_scores.team_names)
    
    @staticmethod
    def from_match_scores(match_scores: MatchScores) -> RatingEngine:
        return RatingEngine(match_scores, match_scores.team_names)
    
    def _get_num_games(self) -> Dict[str, int]:
        result = {}
        for team in self._team_names:
            result[team] = 0
        for m in self._match_scores._scores:
            team_1_number, team_2_number, _, _ = m
            result[self._team_names[team_1_number]] += 1
            result[self._team_names[team_2_number]] += 1
        return result

    def _coeff_rows_from_match(
        self,
        num_games: Dict[str, int],
        team_names: List[str],
        match_row: Tuple[int, int, int, int]
    ) -> List[NDArray[np.float64]]:
        num_teams = len(num_games.keys())
        team_1_number, team_2_number, team_1_score, team_2_score = match_row
        assert team_1_number != team_2_number

        team_1_name = team_names[team_1_number]
        team_2_name = team_names[team_2_number]

        total_score = team_1_score + team_2_score
        match_scalings = [1/np.sqrt(num_games[team_1_name]), 1/np.sqrt(num_games[team_2_name])]

        result = []
        for scal in match_scalings:
            score_items = [(team_1_number, team_1_score), (team_2_number, team_2_score)]
            sorted_items = score_items.sort(lambda x: x[0])
            n1 = sorted_items[0][0]
            n2 = sorted_items[1][0]
            if team_1_score == 0 and team_2_score == 0:
                s1 = 0.5 * scal
                s2 = 0.5 * scal
            else:
                s1 = (sorted_items[1][0]/total_score) * scal
                s2 = (sorted_items[1][1]/total_score) * scal
            
            result.append(np.concatenate([
                np.zeros(n1),
                [s2],
                np.zeros(n2 - n1 - 1),
                [-s1],
                np.zeros(num_teams - n2 - 1)
            ]))
        return result

    def _generate_coeff_matrix(self, team_names: List[str], match_scores: MatchScores) -> NDArray:
        num_games = self._get_num_games()
        match_rows = [
            self._coeff_rows_from_match(num_games, team_names, m) for m in match_scores._scores
        ]

        all_rows: List[NDArray] = []
        for row_list in match_rows:
            all_rows.extend(row_list)
        return np.array([ *all_rows ])
    
    def compute_ratings(self, avg=100) -> Ratings:
        num_teams = len(self._team_names)
        # TODO: drop linearly dependent columns
        C = self._generate_coeff_matrix(self._team_names, self._match_scores)

        CTC = np.dot(C.T, C)
        constrained = np.block([[CTC, np.ones((num_teams, 1))], [np.ones((1, num_teams)), 0]])
        rhs = np.vstack((np.zeros((num_teams, 1)), 1))

        coeffs = scipy.linalg.solve(constrained, rhs)
        result: Dict[str, float] = {}
        for i, [coeff] in enumerate(list(coeffs[:-1])):
            result[self._team_names[i]] = coeff * avg * num_teams
        return Ratings(result)

    def k_fold(self, k=5, random_state=42, alpha=0.05):
        # perform k-fold cross validation
        kf = KFold(
            n_splits=k,
            shuffle=True,
            random_state=random_state
        )
        num_rows = self._match_scores._scores.shape[0]
        num_teams = self._match_scores._scores.shape[1]

        # list of errors on sigma^2 for each test fold
        err_vars = []
        for train_indices, _ in kf.split(self._match_scores.scores):
            test_indices = [i for i in range(num_rows) if i not in train_indices]

            train_match_scores = self._match_scores.pick_matches(train_indices)
            test_match_scores = self._match_scores.pick_matches(test_indices)
        
            engine = RatingEngine.from_match_scores(train_match_scores)
            ratings = engine.compute_ratings()

            # manually compute the error given our ratings
            pred_scores = []
            real_scores = []
            for row in test_match_scores.scores:
                team_1_index, team_2_index, team_1_score, team_2_score = tuple(row)
                team_1_name = self._team_names[team_1_index]
                team_2_name = self._team_names[team_2_index]

                team_1_rating = ratings.get(team_1_name)
                team_2_rating = ratings.get(team_2_name)

                if not team_1_rating or not team_2_rating:
                    continue

                s1_pred = team_1_rating/(team_1_rating + team_2_rating)
                s2_pred = 1 - s1_pred
                pred_scores.extend([s1_pred, s2_pred])            

                s1_real = team_1_score/(team_1_score + team_2_score)
                s2_real = 1 - s1_real
                real_scores.extend([s1_real, s2_real])
            
            # relative percent difference based formalism for RSS
            errs = np.array([
                np.abs(x - y)/(np.abs(x) + np.abs(y))
                for x, y in zip(pred_scores, real_scores)
            ])
            RSS = np.sum(errs**2)
            err_vars.append(
                RSS/(num_rows - num_teams - 1)
            )
        
        return 1 - np.sqrt(np.mean(err_vars))
    
    def win_accuracy(self, ratings=None):
        if not ratings:
            ratings = self.compute_ratings()

        correct = 0
        for row in self._match_scores.scores:
            team_1_index, team_2_index, team_1_score, team_2_score = row
            team_1_name = self._team_names[team_1_index]
            team_2_name = self._team_names[team_2_index]

            team_1_rating = ratings.get(team_1_name)
            team_2_rating = ratings.get(team_2_name)

            if team_1_rating < team_2_rating and team_1_score < team_2_score:
                correct += 1
            if team_2_rating < team_1_rating and team_2_score < team_1_score:
                correct += 1
        return correct / self._match_scores.scores.shape[0]
