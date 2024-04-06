# [1]
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from time import strptime, mktime, ctime
import re
from time import strptime, mktime
from dateutil.parser import isoparse
import datetime


# [2]
dataset_datetime = isoparse("2021-04-30")
print(dataset_datetime)
# Read Data
data_raw = pd.read_csv("tracks.csv", low_memory=False)
data_raw.describe()


# [3]
# Filter Relevant Columns
wanted_columns = [
    "popularity",
    "duration_ms",
    "explicit",
    "release_date",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]
processed_data = data_raw.loc[:, wanted_columns]


# [4]
# Convert "release_date" timestamps to "days_since_release"

newest = datetime.datetime(1, 1, 1)
oldest = datetime.datetime(2023, 1, 1)

datetimes = []
times_since_release = []
for row_idx, row in processed_data.iterrows():
    datetime_str = str(row["release_date"])

    if re.search("^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$", datetime_str):
        pass
    elif re.search("^[0-9][0-9][0-9][0-9]$", datetime_str):
        datetime_str += "-06-15"
    elif re.search("^[0-9][0-9][0-9][0-9]-[0-9][0-9]$", datetime_str):
        datetime_str += "-15"
    else:
        raise ValueError

    release_datetime = isoparse(datetime_str)
    time_since_release = (dataset_datetime - release_datetime).total_seconds()

    # print(datetime_str)
    newest = max(newest, release_datetime)
    oldest = min(oldest, release_datetime)

    if time_since_release < 0:
        print(f"{row['name']}, {release_datetime}, {datetime_str}")

    times_since_release.append(time_since_release / (60 * 60 * 24))
processed_data["days_since_release"] = times_since_release
print(f"Newest: {newest}\nOldest: {oldest}")
processed_data.describe()


# [5]
processed_data.drop("release_date", axis=1, inplace=True)


# [6]
# Remove rows with empty values
processed_data.replace(["", None], np.nan, inplace=True)
processed_data.dropna(inplace=True)

# Display the distribution of popularity
sns.displot(processed_data["popularity"], kde=True)
processed_data.reset_index(drop=True, inplace=True)
print(processed_data.shape)
processed_data.describe()


# [7]
"""#Remove Outliers
processed_data = processed_data[processed_data["popularity"] < 50000]

sns.displot(processed_data["popularity"], kde=True)
print(processed_data.shape)
processed_data.describe()"""


# [8]
# Normalize the features
processed_data = processed_data.astype("float64")
processed_data.iloc[:, 1:] = (
    processed_data.iloc[:, 1:] - processed_data.iloc[:, 1:].mean()
) / processed_data.iloc[
    :, 1:
].std()  # z-score standardization w/ mean:0 std:1
# processed_data.iloc[:, 1:] = (processed_data-processed_data.min()) / (processed_data.max()-processed_data.min()) # min-max normalization w/ min:0 max:1
processed_data.replace(["", None, np.nan], 0, inplace=True)


processed_data.reset_index(drop=True, inplace=True)
print(processed_data.shape)
processed_data.describe()


# [9]
# Save as another csv file
processed_data.to_csv("processed_database_2.csv", index=False)
