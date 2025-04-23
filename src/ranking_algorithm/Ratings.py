from typing import Dict, Union

from sklearn.model_selection import KFold

class Ratings():
    def __init__(self, rating_dict: Dict[str, float]):
        self._rating_dict = rating_dict
    
    def get(self, team_name: str) -> Union[None, float]:
        if team_name not in self._rating_dict:
            return None
        return self._rating_dict[team_name]
    
    @property
    def rating_dict(self):
        return self._rating_dict