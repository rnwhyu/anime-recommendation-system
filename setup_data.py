import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = "./data"

ANIME_CLEANED = "{}/anime_cleaned.csv".format(DATA_DIR)
USER_CLEANED = "{}/users_cleaned.csv".format(DATA_DIR)
ANIMELIST_CLEANED = "{}/animelists_cleaned.csv".format(DATA_DIR)

USER_CLEANED_1000 = "{}/users_cleaned_1000.csv".format(DATA_DIR)
PREPROCESSED_ANIME = "{}/preprocessed_anime.csv".format(DATA_DIR)
PREPROCESSED_INPUT_1000 = "{}/preprocessed_input_1000.csv".format(DATA_DIR)
USER_ANIME_RATING_MATRIX = "{}/user_anime_rating_matrix.csv".format(DATA_DIR)

user_info_cols = ['username', 'user_id']
user_info = pd.read_csv(USER_CLEANED, usecols=user_info_cols)
user = user_info.dropna()
print(user.shape, '\n', user.head())

del user_info

print('\n\n\n')

anime_info_cols = [
    'anime_id', 'title', 'title_english', 'score', 'scored_by', 'rank',
    'popularity', 'members', 'favorites', 'genre'
]
anime_info = pd.read_csv(ANIME_CLEANED, usecols=anime_info_cols)
anime = anime_info.dropna()
print(anime.shape, '\n', anime.head())

del anime_info

anime_genres = anime.filter('genre')
anime_genres['genre'] = anime['genre'].str.split(',')
list_of_genre = anime_genres.explode('genre')
list_of_genre = list_of_genre.drop_duplicates('genre').filter(['genre'])
list_of_genre = [genre.strip() for genre in list_of_genre['genre'].unique()]
list_of_genre = list(set(list_of_genre))

anime_lists_cols = ['username', 'anime_id', 'my_score']
anime_lists_dtype = {
    "username": "string",
    "anime_id": "uint16",
    "my_score": "int8"
}
anime_lists = pd.read_csv(ANIMELIST_CLEANED,
                          usecols=anime_lists_cols,
                          dtype=anime_lists_dtype)
users_anime = anime_lists.dropna()
print(users_anime.head())

del anime_lists

print(anime.info(verbose=False, memory_usage='deep'), '\n')
print(user.info(verbose=False, memory_usage='deep'), '\n')
print(users_anime.info(verbose=False, memory_usage='deep'), '\n')

user_with_anime_ratings = pd.merge(user,
                                   users_anime,
                                   on=['username',
                                       'username']).drop(columns=['username'])
user_anime_rating_counts = pd.DataFrame(
    user_with_anime_ratings.groupby('user_id')['my_score'].count())

# remove if < 500
filtered_user_anime_rating_counts = user_anime_rating_counts[
    user_anime_rating_counts['my_score'] >= 500]

user = user[user['user_id'].isin(filtered_user_anime_rating_counts.index)]
user = user.sample(1000)
print(user.shape, '\n', user.head())
print()

preprocessed_anime = anime.filter(['anime_id', 'genre'])

anime_with_ratings = pd.merge(preprocessed_anime,
                              users_anime,
                              on=['anime_id', 'anime_id'])
anime_rating_counts = pd.DataFrame(
    anime_with_ratings.groupby('anime_id')['my_score'].count())

# remove if < 1000
filtered_anime_rating_counts = anime_rating_counts[
    anime_rating_counts['my_score'] >= 1000]

preprocessed_anime = preprocessed_anime[preprocessed_anime['anime_id'].isin(
    filtered_anime_rating_counts.index)]
print(preprocessed_anime.shape, '\n', preprocessed_anime.head())

# preparing dataset for model training
preprocessed_input = pd.merge(user, users_anime,
                              on=['username',
                                  'username']).drop(columns=['username'])

del users_anime
gc.collect()

preprocessed_input = pd.merge(preprocessed_input,
                              preprocessed_anime,
                              on=['anime_id', 'anime_id'])
preprocessed_input = preprocessed_input.drop(
    preprocessed_input[preprocessed_input.my_score == 0].index)

for genre in list_of_genre:
    preprocessed_anime[genre] = np.where(
        preprocessed_anime['genre'].str.contains(genre), 1, 0)

    preprocessed_input[genre] = np.where(
        preprocessed_input['genre'].str.contains(genre), 1, 0)

preprocessed_input['label'] = np.where(preprocessed_input['my_score'] > 5,
                                       'Suka', 'Tidak Suka')
preprocessed_input = preprocessed_input.drop(columns=['genre'])
preprocessed_input = preprocessed_input.drop_duplicates()
preprocessed_input = preprocessed_input.dropna()

print(preprocessed_input.head())

preprocessed_input = preprocessed_input[preprocessed_input['user_id'].isin(
    user['user_id'])]
preprocessed_anime = preprocessed_anime.drop(columns=['genre'])
preprocessed_anime_cols = ['anime_id']
preprocessed_anime_cols = preprocessed_anime_cols + list(
    preprocessed_input.columns)[3:-1]
preprocessed_anime = preprocessed_anime.filter(preprocessed_anime_cols)

print(preprocessed_anime.head())

user_anime_rating_matrix = preprocessed_input.pivot_table(index='user_id',
                                                          columns='anime_id',
                                                          values='my_score')
# replace NaN values with 0
user_anime_rating_matrix = user_anime_rating_matrix.fillna(0)
# display the top few rows
print(user_anime_rating_matrix.head())

user = user[user['user_id'].isin(preprocessed_input['user_id'])]

print('encoding anime id')

anime_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_anime_id = anime_encoder.fit_transform(
    preprocessed_anime['anime_id'].values.reshape(-1, 1))
encoded_anime_id = pd.DataFrame(
    encoded_anime_id,
    columns=[
        "a_{}".format(anime_id.replace('x0_', ''))
        for anime_id in anime_encoder.get_feature_names_out()
    ])
encoded_anime_id.index = preprocessed_anime.index
preprocessed_anime = pd.concat([encoded_anime_id, preprocessed_anime],
                               axis=1).drop(['anime_id'], axis=1)

del anime_encoder
del encoded_anime_id
gc.collect()

print('encoding anime id on input')

anime_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_anime_id = anime_encoder.fit_transform(
    preprocessed_input['anime_id'].values.reshape(-1, 1))
encoded_anime_id = pd.DataFrame(
    encoded_anime_id,
    columns=[
        "a_{}".format(anime_id.replace('x0_', ''))
        for anime_id in anime_encoder.get_feature_names_out()
    ])
encoded_anime_id.index = preprocessed_input.index
preprocessed_input = pd.concat([encoded_anime_id, preprocessed_input],
                               axis=1).drop(['anime_id'], axis=1)

del anime_encoder
del encoded_anime_id
gc.collect()

print('encoding user id')

user_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_user_id = user_encoder.fit_transform(
    preprocessed_input['user_id'].values.reshape(-1, 1))
encoded_user_id = pd.DataFrame(
    encoded_user_id,
    columns=[
        user_id.replace('x0_', '')
        for user_id in user_encoder.get_feature_names_out()
    ])
encoded_user_id.index = preprocessed_input.index
preprocessed_input = pd.concat([encoded_user_id, preprocessed_input],
                               axis=1).drop(['user_id'], axis=1)

del user_encoder
del encoded_user_id
gc.collect()

print(preprocessed_input.head())

print('user_1000', user.shape)
print('preprocessed_input', preprocessed_input.shape)
print('preprocessed_anime', preprocessed_anime.shape)
print('user_anime_rating_matrix', user_anime_rating_matrix.shape)

# export user_1000 to csv
with open(USER_CLEANED_1000, 'w', encoding='utf-8-sig') as f:
    user.to_csv(f, index=False)

# export preprocessed_input_1000 to csv
with open(PREPROCESSED_INPUT_1000, 'w', encoding='utf-8-sig') as f:
    preprocessed_input.to_csv(f, index=False)

# export preprocessed_anime to csv
with open(PREPROCESSED_ANIME, 'w', encoding='utf-8-sig') as f:
    preprocessed_anime.to_csv(f, index=False)

# export user_anime_rating_matrix to csv
with open(USER_ANIME_RATING_MATRIX, 'w', encoding='utf-8-sig') as f:
    user_anime_rating_matrix.to_csv(f, index=True, index_label='user_id')
