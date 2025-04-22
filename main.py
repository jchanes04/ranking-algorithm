import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from load_data import generate_match_scores
from ratings import compute_ratings

def main():
    df = pd.read_csv("data/football2014.csv")
    num_rows = len(df)

    # perform k-fold cross validation
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    kf.split(df)

    mse_list = []
    for train_indices, _ in kf.split(df):
        test_indices = [i for i in range(num_rows) if i not in train_indices]

        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]
        match_scores, team_names = generate_match_scores([row for _, row in train_df.iterrows()])
    
        ratings = compute_ratings(match_scores, team_names)

        # manually compute the error given our ratings
        sq_errors = []
        for _, row in test_df.iterrows():
            team_1_rating = ratings[row["team1Name"]]
            team_2_rating = ratings[row["team2Name"]]

            s1_pred = team_1_rating/(team_1_rating + team_2_rating)

            s1_real = row["team1Score"]/(row["team1Score"] + row["team2Score"])
            sq_errors.append((s1_pred - s1_real)**2)
        mse_list.append(np.average(sq_errors))


    results = [(name, rating) for name, rating in ratings.items()]
    results.sort(key=lambda x: x[1])
    # for name, rating in results:
    #     print(f"{name}: {rating}")
    # print(mse_list)
    avg_mse = np.average(mse_list)
    print(f"average MSE: {avg_mse}")
    print(f"expected error: {np.sqrt(avg_mse)}")    


if __name__ == "__main__":
    main()
