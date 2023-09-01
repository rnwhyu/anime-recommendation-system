from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import LeaveOneOut
from random import choice, sample
import pandas as pd
import operator
import pickle

from app.model.anime import (filter_anime_by_indices, get_all_anime_genre,
                             get_all_anime_id, get_all_preprocessed_anime)
from app.model.user import get_all_preprocessed_user

USER_ANIME_RATING_MATRIX = "./data/user_anime_rating_matrix.csv"

CLASSIFIER_MODEL = "./export/classifier_model.pkl"
# SCALER = "./export/scaler.pkl"


class Recommender:

    def __init__(self):
        with open(CLASSIFIER_MODEL, 'rb') as f:
            model = pickle.load(f)

        if model is None:
            print('Model not found')
            exit()

        self.model = model

        # with open(SCALER, 'rb') as f:
        #     scaler = pickle.load(f)

        # if scaler is None:
        #     print('Scaler not found')
        #     exit()

        # self.scaler = scaler

        anime_ids = get_all_anime_id()
        user_anime_rating_matrix_dtype = {"user_id": "int32"}
        for anime_id in anime_ids:
            user_anime_rating_matrix_dtype[anime_id] = "float16"

        user_anime_rating_matrix = pd.read_csv(
            USER_ANIME_RATING_MATRIX,
            dtype=user_anime_rating_matrix_dtype,
            index_col=0)

        self.matrix = user_anime_rating_matrix

    def predict_user_dislikes(self, user_id):
        data = get_all_preprocessed_anime()
        user = get_all_preprocessed_user(data)
        user[str(user_id)] = 1
        data = pd.concat([user, data], axis=1)
        # data['user_id'] = int(user_id)
        # cols = data.columns.tolist()
        # cols = cols[-1:] + cols[:-1]
        # data = data[cols]

        X = data
        # X = self.scaler.transform(data)

        y_pred = self.model.predict(X)
        data['label'] = y_pred

        # get all anime_id of negative label in data
        disliked_anime = data[data['label'] == 'Tidak Suka']
        disliked_anime = disliked_anime.drop(columns=user.columns, axis=1)
        disliked_anime = disliked_anime.drop(columns=get_all_anime_genre(),
                                             axis=1)
        disliked_anime_indices = disliked_anime.stack().reset_index(
            name='val').query('val == 1').groupby('level_0')['level_1'].apply(
                list)

        disliked_anime_indices = [
            index[0].replace('a_', '') for index in disliked_anime_indices
        ]

        return disliked_anime_indices

    def similar_users(self,
                      user_id,
                      k=5,
                      disliked_anime_ids=[],
                      leave_one_out_batch=False):
        # filter out animes which are predicted to be disliked by the user
        dropped_matrix = self.matrix.copy()
        if len(disliked_anime_ids) == self.matrix.shape[1]:
            disliked_ids = sample(disliked_anime_ids, round(len(disliked_anime_ids) / 2))
            dropped_matrix = self.matrix.drop(disliked_ids, axis=1)
        else:
            dropped_matrix = self.matrix.drop(disliked_anime_ids, axis=1)

        # create a df of just the current user
        user = dropped_matrix[dropped_matrix.index == user_id]
        # transpose it so its easier to filter
        user_df_transposed = user.transpose()
        # rename the column as 'rating'
        user_df_transposed.columns = ['rating']
        # generate a list of animes the user has seen
        animes_seen = user_df_transposed[(user_df_transposed['rating'] != 0) & ((user_df_transposed['rating'] >= 10) | (user_df_transposed['rating'] <= 4))]
        animes_seen_list = animes_seen.index.tolist()
        if len(animes_seen_list) == 0:
            animes_seen = user_df_transposed[(user_df_transposed['rating'] != 0) & (user_df_transposed['rating'] <= 5)]
            animes_seen_list = animes_seen.index.tolist()
            if len(animes_seen_list) == 0:
                animes_seen = user_df_transposed[(user_df_transposed['rating'] != 0) & (user_df_transposed['rating'] >= 9)]
                animes_seen_list = animes_seen.index.tolist()
                if len(animes_seen_list) == 0:
                    animes_seen = user_df_transposed[(user_df_transposed['rating'] != 0) & ((user_df_transposed['rating'] >= 8) | (user_df_transposed['rating'] <= 5))]
                    animes_seen_list = animes_seen.index.tolist()
                    if len(animes_seen_list) == 0:
                        animes_seen = user_df_transposed[user_df_transposed['rating'] != 0]
                        animes_seen_list = animes_seen.index.tolist()

        train_index = [index for index in range(len(animes_seen_list))]
        test_index = []
        try:
            test_index = choice(list(enumerate(animes_seen_list)))[0]
        except IndexError:
            test_index = 0
        test_index = [test_index]
        splits = [(train_index, test_index)]

        if leave_one_out_batch:
            leave_one_out = LeaveOneOut()
            splits = leave_one_out.split(animes_seen_list)

        users = list()
        for train_index, test_index in splits:
            # and a df of all other users
            similarities = list()
            other_users = dropped_matrix[dropped_matrix.index != user_id]
            if len(test_index) > 0:
                current_user = user.copy()
                current_user.loc[user_id, animes_seen_list[test_index[0]]] = 0

                # calc cosine similarity between user and each other user
                similarities = cosine_similarity(current_user,
                                                 other_users)[0].tolist()
            else:
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
            users.append(([u[0] for u in top_users_similarities],
                          [animes_seen_list[index] for index in test_index]))

        return users, dropped_matrix

    def get_anime_recommendation(self,
                                 user_id,
                                 similar_users_count=5,
                                 items=10,
                                 use_model=True,
                                 test_all_anime=False):
        # predict if use model
        disliked_anime_indices = list()
        if use_model:
            disliked_anime_indices = self.predict_user_dislikes(user_id)
            # # print disliked anime
            # anime_information = filter_anime_by_indices(
            #     [int(pred) for pred in disliked_anime_indices])
            # print(anime_information)

        similar_users_, dropped_matrix = self.similar_users(
            user_id,
            k=similar_users_count,
            disliked_anime_ids=disliked_anime_indices,
            leave_one_out_batch=test_all_anime)

        recommended_anime_information = list()
        seen_anime_information = list()
        leave_one_out_result = list()
        for similar_user_indices, leave_one_out_anime in similar_users_:
            # load vectors for similar users
            similar_users = dropped_matrix[dropped_matrix.index.isin(
                similar_user_indices)]
            # calc avg ratings across the 3 similar users5
            similar_users = similar_users.mean(axis=0)
            # convert to dataframe so its easy to sort and filter
            similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
            # load vector for the current user
            user_df = dropped_matrix[dropped_matrix.index == user_id]
            user_df.loc[user_id, leave_one_out_anime] = 0

            # transpose it so its easier to filter
            user_df_transposed = user_df.transpose()
            # rename the column as 'rating'
            user_df_transposed.columns = ['rating']
            # generate a list of animes the user has not seen
            animes_unseen = user_df_transposed[user_df_transposed['rating'] ==
                                               0].index.tolist()
            animes_seen = user_df_transposed[user_df_transposed['rating'] != 0]
            animes_seen_list = [int(leave_one_out_anime[0])]
            animes_seen_list.extend(animes_seen.index.astype(int).tolist())

            # filter avg ratings of similar users for only anime the current
            # user has not seen
            similar_users_df_filtered = similar_users_df[
                similar_users_df.index.isin(animes_unseen)]
            # order the dataframe
            similar_users_df_ordered = similar_users_df_filtered.sort_values(
                by=['mean'], ascending=False)

            # grab the top n anime
            top_n_anime = similar_users_df_ordered.head(items)
            top_n_anime.index = top_n_anime.index.astype(int)
            top_n_anime_indices = top_n_anime.index.tolist()
            # lookup these anime in the other dataframe to find names

            anime_information = filter_anime_by_indices(top_n_anime_indices)
            anime_information = anime_information.join(top_n_anime,
                                                       on="anime_id")
            anime_information['mean'] = [
                "{:.1f}".format(i) for i in anime_information['mean'].values
            ]

            is_leave_one_out_exist = not anime_information[
                anime_information["anime_id"] == int(
                    leave_one_out_anime[0])].empty
            is_leave_one_out_liked = self.matrix.loc[
                user_id, leave_one_out_anime].values[0] > 5

            result_of_leave_one_out = ''
            if is_leave_one_out_liked:
                if is_leave_one_out_exist:
                    result_of_leave_one_out = 'TP'
                else:
                    result_of_leave_one_out = 'FN'
            else:
                if is_leave_one_out_exist:
                    result_of_leave_one_out = 'FP'
                else:
                    result_of_leave_one_out = 'TN'

            seen_anime = filter_anime_by_indices(animes_seen_list)
            seen_anime['is_leave_one_out'] = [
                anime_id == int(leave_one_out_anime[0])
                for anime_id in seen_anime['anime_id']
            ]
            seen_anime['user_rating'] = [
                'Liked'
                if self.matrix.loc[user_id, str(anime_id)] > 5 else 'Disliked'
                for anime_id in seen_anime['anime_id']
            ]

            anime_information['is_leave_one_out'] = [
                anime_id == int(leave_one_out_anime[0])
                for anime_id in anime_information['anime_id']
            ]

            recommended_anime_information.append(anime_information)
            seen_anime_information.append(seen_anime)
            leave_one_out_result.append(result_of_leave_one_out)

        if test_all_anime:
            return leave_one_out_result

        return recommended_anime_information[0], seen_anime_information[
            0], leave_one_out_result[0]
