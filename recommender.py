from sklearn.metrics.pairwise import cosine_similarity
from pandas.api.types import CategoricalDtype
import pandas as pd
import numpy as np
import operator
import pickle

DATA_DIR = "./data"
EXPORT_DIR = "./export"

USER_CLEANED = "{}/users_cleaned_1000.csv".format(DATA_DIR)
ANIME_CLEANED = "{}/anime_cleaned.csv".format(DATA_DIR)
PREPROCESSED_ANIME = "{}/preprocessed_anime.csv".format(DATA_DIR)
PREPROCESSED_INPUT = "{}/preprocessed_input_1000.csv".format(DATA_DIR)
USER_ANIME_RATING_MATRIX = "{}/user_anime_rating_matrix.csv".format(DATA_DIR)

CLASSIFIER_MODEL = "{}/classifier_model.pkl".format(EXPORT_DIR)
SCALER = "{}/scaler.pkl".format(EXPORT_DIR)

with open(CLASSIFIER_MODEL, 'rb') as f:
    model = pickle.load(f)

if model is None:
    print('Model not found')
    exit()

with open(SCALER, 'rb') as f:
    scaler = pickle.load(f)

if scaler is None:
    print('Scaler not found')
    exit()

user_cols = ['username', 'user_id']
user = pd.read_csv(USER_CLEANED, usecols=user_cols)

anime_cols = [
    'anime_id', 'title', 'title_english', 'score', 'scored_by', 'rank',
    'popularity', 'members', 'favorites', 'genre'
]
anime = pd.read_csv(ANIME_CLEANED, usecols=anime_cols).dropna()
print(anime.shape, '\n', anime.head())


def filter_anime_by_indices(indices):
    df = anime[anime['anime_id'].isin(indices)]

    categorical_indices_order = CategoricalDtype(indices, ordered=True)
    df['anime_id'] = df['anime_id'].astype(categorical_indices_order)
    df = df.sort_values(['anime_id'])
    return df


anime_genres = anime.filter('genre')
anime_genres['genre'] = anime['genre'].str.split(',')
list_of_genre = anime_genres.explode('genre')
list_of_genre = list_of_genre.drop_duplicates('genre').filter(['genre'])
list_of_genre = [genre.strip() for genre in list_of_genre['genre'].unique()]
list_of_genre = list(set(list_of_genre))

preprocessed_anime_dtype = {"label": "category"}

for anime_id in anime['anime_id']:
    preprocessed_anime_dtype["a_{}".format(str(anime_id))] = "int8"

for genre in list_of_genre:
    preprocessed_anime_dtype[genre] = "int8"

preprocessed_anime = pd.read_csv(PREPROCESSED_ANIME,
                                 dtype=preprocessed_anime_dtype)

print(preprocessed_anime.shape, '\n', preprocessed_anime.head())

user_anime_rating_matrix_dtype = {"user_id": "int32"}
for anime_id in anime['anime_id']:
    user_anime_rating_matrix_dtype[str(anime_id)] = "float16"

user_anime_rating_matrix = pd.read_csv(USER_ANIME_RATING_MATRIX,
                                       dtype=user_anime_rating_matrix_dtype,
                                       index_col=0)

print(user_anime_rating_matrix.shape, '\n', user_anime_rating_matrix.head())

print(anime.info(verbose=False, memory_usage='deep'), '\n')
print(user_anime_rating_matrix.info(verbose=False, memory_usage='deep'), '\n')


def predict_user_dislikes(user_id):
    data = preprocessed_anime.copy()
    preprocessed_user = pd.DataFrame(
        np.zeros((data.shape[0], user.shape[0])),
        columns=[
            str(user_id) for user_id in user.sort_values('user_id')['user_id']
        ],
        dtype='int8')
    preprocessed_user[str(user_id)] = 1
    data = pd.concat([preprocessed_user, data], axis=1)

    X = data

    y_pred = model.predict(X)
    data['label'] = y_pred

    # get all anime_id of negative label in data
    disliked_anime = data[data['label'] == 'Tidak Suka']
    disliked_anime = disliked_anime.drop(columns=preprocessed_user.columns,
                                         axis=1)
    disliked_anime = disliked_anime.drop(columns=list_of_genre, axis=1)
    disliked_anime_indices = disliked_anime.stack().reset_index(
        name='val').query('val == 1').groupby('level_0')['level_1'].apply(list)

    disliked_anime_indices = [
        index[0].replace('a_', '') for index in disliked_anime_indices
    ]

    return disliked_anime_indices


def similar_users(user_id, matrix, k=3):
    # create a df of just the current user
    user = matrix[matrix.index == user_id]
    # and a df of all other users
    other_users = matrix[matrix.index != user_id]
    # calc cosine similarity between user and each other user
    similarities = cosine_similarity(user, other_users)[0].tolist()
    # create list of indices of these users
    indices = other_users.index.tolist()
    # create key/values pairs of user index and their similarity
    index_similarity = dict(zip(indices, similarities))
    # sort by similarity
    index_similarity_sorted = sorted(index_similarity.items(),
                                     key=operator.itemgetter(1))
    index_similarity_sorted.reverse()
    # grab k users off the top
    top_users_similarities = index_similarity_sorted[:k]
    users = [u[0] for u in top_users_similarities]
    return users


def recommend_item(user_index,
                   similar_user_indices,
                   matrix,
                   items=5,
                   use_model=False):
    # load vectors for similar users
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    # calc avg ratings across the 3 similar users
    similar_users = similar_users.mean(axis=0)
    # convert to dataframe so its easy to sort and filter
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    # load vector for the current user
    user_df = matrix[matrix.index == user_index]
    # transpose it so its easier to filter
    user_df_transposed = user_df.transpose()
    # rename the column as 'rating'
    user_df_transposed.columns = ['rating']
    # remove any rows without a 0 value. Anime not watched yet
    user_df_transposed = user_df_transposed[user_df_transposed['rating'] == 0]
    # generate a list of animes the user has not seen
    animes_unseen = user_df_transposed.index.tolist()
    # filter avg ratings of similar users for only anime the current
    # user has not seen
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(
        animes_unseen)]
    # order the dataframe
    similar_users_df_ordered = similar_users_df_filtered.sort_values(
        by=['mean'], ascending=False)
    # predict if use model
    if use_model:
        disliked_anime_indices = predict_user_dislikes(current_user)
        # filter out animes which are predicted to be disliked by the user
        # print(similar_users_df_ordered.shape)
        similar_users_df_ordered = similar_users_df_ordered.drop(
            index=disliked_anime_indices, errors='ignore')
        # print(similar_users_df_ordered.shape)

        # print disliked anime
        anime_information = filter_anime_by_indices(
            [int(pred) for pred in disliked_anime_indices])
        # print(anime_information)

    # grab the top n anime
    top_n_anime = similar_users_df_ordered.head(items)
    top_n_anime_indices = top_n_anime.index.tolist()
    top_n_anime_indices = [int(anime_id) for anime_id in top_n_anime_indices]
    # print(top_n_anime_indices)
    # lookup these anime in the other dataframe to find names

    anime_information = filter_anime_by_indices(top_n_anime_indices)
    return anime_information  # items


# preprocessed_input_dtype = {"my_score": "int8", "label": "category"}
# for genre in list_of_genre:
#     preprocessed_input_dtype[genre] = "int8"

# for user_id in user['user_id']:
#     preprocessed_input_dtype[str(user_id)] = "int8"

# for anime_id in anime['anime_id']:
#     preprocessed_input_dtype["a_{}".format(str(anime_id))] = "int8"

# preprocessed_input = pd.read_csv(
#     PREPROCESSED_INPUT, dtype=preprocessed_input_dtype).reset_index(drop=True)

# try it out
current_user = 459521

# print(np.count_nonzero(user_anime_rating_matrix))
# user_anime_rating_matrix.loc[
#     current_user,
#     user_anime_rating_matrix.sample(1).columns] = 0
# print(np.count_nonzero(user_anime_rating_matrix))
from random import sample
leave_one_out_anime = sample([user_anime_rating_matrix[user_anime_rating_matrix.index == current_user].columns.values[x].tolist() for x in user_anime_rating_matrix[user_anime_rating_matrix.index == current_user].gt(0.0).values][0], 1)[0]
print(leave_one_out_anime)
print(user_anime_rating_matrix[user_anime_rating_matrix.index == current_user][leave_one_out_anime])
user_anime_rating_matrix.loc[current_user,leave_one_out_anime] = 0
print(user_anime_rating_matrix[user_anime_rating_matrix.index == current_user][leave_one_out_anime])

similar_user_indices = similar_users(current_user, user_anime_rating_matrix, 5)
recommendation = recommend_item(current_user,
                                similar_user_indices,
                                user_anime_rating_matrix,
                                100,
                                use_model=True)
print(similar_user_indices)
print(recommendation["title"])
print(recommendation[recommendation["anime_id"] == int(leave_one_out_anime)])

# for anime_id in recommendation['anime_id']:
#     filtered_label = preprocessed_input[
#         (preprocessed_input[str(current_user)] == 1)
#         & (preprocessed_input['a_' + str(anime_id)] == 1)].label

#     filtered_label = 'Unidentified' if filtered_label.empty else (
#         filtered_label.item())
#     print(filtered_label)

# current_user = 6335523

# print(np.count_nonzero(user_anime_rating_matrix))
# user_anime_rating_matrix.loc[
#     current_user,
#     user_anime_rating_matrix.sample(frac=0.8, axis=1).columns] = 0
# print(np.count_nonzero(user_anime_rating_matrix))

# similar_user_indices = similar_users(current_user, user_anime_rating_matrix, 3)
# recommendation = recommend_item(current_user,
#                                 similar_user_indices,
#                                 user_anime_rating_matrix,
#                                 use_model=True)

# print(recommendation)

# for anime_id in recommendation['anime_id']:
#     filtered_label = preprocessed_input[
#         (preprocessed_input[str(current_user)] == 1)
#         & (preprocessed_input['a_' + str(anime_id)] == 1)].label

#     filtered_label = 'Unidentified' if filtered_label.empty else (
#         filtered_label.item())
#     print(filtered_label)

# current_user = 4611

# print(np.count_nonzero(user_anime_rating_matrix))
# user_anime_rating_matrix.loc[
#     current_user,
#     user_anime_rating_matrix.sample(frac=0.8, axis=1).columns] = 0
# print(np.count_nonzero(user_anime_rating_matrix))

# similar_user_indices = similar_users(current_user, user_anime_rating_matrix, 3)
# recommendation = recommend_item(current_user,
#                                 similar_user_indices,
#                                 user_anime_rating_matrix,
#                                 use_model=True)

# print(recommendation)

# for anime_id in recommendation['anime_id']:
#     filtered_label = preprocessed_input[
#         (preprocessed_input[str(current_user)] == 1)
#         & (preprocessed_input['a_' + str(anime_id)] == 1)].label

#     filtered_label = 'Unidentified' if filtered_label.empty else (
#         filtered_label.item())
#     print(filtered_label)
