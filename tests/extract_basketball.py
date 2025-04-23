import pandas as pd
import random

df = pd.read_csv("data/basketball.csv")

def season_year_from_date(date_str: str):
    date_year = int(date_str.split("-")[0])
    if int(date_str.split("-")[1]) <= 7:
        return date_year - 1
    else:
        return date_year

for i in range(20):
    year = random.choice(range(1980,2025))
    filtered = df.iloc[[i for i, date_str in enumerate(df["gameDate"]) if season_year_from_date(date_str) == year]]
    
    year_data = {
        "team1Name": filtered["hometeamName"],
        "team2Name": filtered["awayteamName"],
        "team1Score": filtered["homeScore"],
        "team2Score": filtered["awayScore"]
    }
    year_df = pd.DataFrame.from_dict(year_data)
    year_df.to_csv(f"data/basketball{year}.csv")