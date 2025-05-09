import pandas as pd
import numpy as np
from datetime import datetime
import os

# Define paths with absolute references
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_PATH = os.path.join(SCRIPT_DIR, 'raw/ml-1m/')
PROCESSED_DATA_PATH = os.path.join(SCRIPT_DIR, 'processed/')

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

print("Loading MovieLens 1M dataset...")
# Load datasets
# Ratings: UserID::MovieID::Rating::Timestamp
r_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings = pd.read_csv(os.path.join(BASE_DATA_PATH, 'ratings.dat'), sep='::', names=r_cols, engine='python', encoding='latin-1')

# Movies: MovieID::Title::Genres
m_cols = ['MovieID', 'Title', 'Genres']
movies = pd.read_csv(os.path.join(BASE_DATA_PATH, 'movies.dat'), sep='::', names=m_cols, engine='python', encoding='latin-1')

# Users: UserID::Gender::Age::Occupation::Zip-code
u_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
users = pd.read_csv(os.path.join(BASE_DATA_PATH, 'users.dat'), sep='::', names=u_cols, engine='python', encoding='latin-1')

print(f"Ratings: {ratings.shape}, Movies: {movies.shape}, Users: {users.shape}")

print("Cleaning and converting data types...")
# Check for missing values
print(f"Missing values - Ratings: {ratings.isnull().sum().sum()}, Movies: {movies.isnull().sum().sum()}, Users: {users.isnull().sum().sum()}")

# Convert Timestamp to datetime
ratings['Timestamp'] = ratings['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))

print("Processing movie genres...")
# Extract and one-hot encode genres
movies['Genres_List'] = movies['Genres'].apply(lambda x: x.split('|'))
all_genres = sorted(list(set([genre for sublist in movies['Genres_List'] for genre in sublist])))
print(f"Unique genres: {all_genres}")

for genre in all_genres:
    movies[f'Genre_{genre.replace("-", "_")}'] = movies['Genres_List'].apply(lambda x: 1 if genre in x else 0)

movies_processed = movies.copy()

print("Simulating contextual features...")
# Time of Day from Timestamp
def get_time_of_day(timestamp):
    hour = timestamp.hour
    if 5 <= hour < 12:  # Morning: 5 AM - 11:59 AM
        return 'morning'
    elif 12 <= hour < 17:  # Afternoon: 12 PM - 4:59 PM
        return 'afternoon'
    elif 17 <= hour < 21:  # Evening: 5 PM - 8:59 PM
        return 'evening'
    else:  # Night: 9 PM - 4:59 AM
        return 'night'

ratings['Time_of_Day'] = ratings['Timestamp'].apply(get_time_of_day)

# Device Type (Random Assignment)
device_types = ['mobile', 'tablet', 'desktop', 'TV']
np.random.seed(42)  # for reproducibility
ratings['Device_Type'] = np.random.choice(device_types, size=len(ratings))

print("Merging datasets...")
# Merge ratings with users
df = pd.merge(ratings, users, on='UserID', how='left')

# Merge with movies (with one-hot encoded genres)
df = pd.merge(df, movies_processed.drop(columns=['Genres_List']), on='MovieID', how='left')

print(f"Final dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Save the processed data
output_file = os.path.join(PROCESSED_DATA_PATH, 'processed_movielens_data.csv')
df.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}") 