import pandas as pd
import numpy as np

USER_CLEANED_CSV = './data/users_cleaned_1000.csv'
user = pd.read_csv(USER_CLEANED_CSV,
                   usecols=['username', 'user_id'],
                   dtype={
                       'username': 'string',
                       'user_id': 'int32'
                   })


# get all user as json
def get_all_user():
    return user.rename(columns={
        'username': 'text',
        'user_id': 'id'
    }).to_json(orient='records')


def get_all_preprocessed_user(preprocessed_anime):
    return pd.DataFrame(
        np.zeros((preprocessed_anime.shape[0], user.shape[0])),
        columns=[
            str(user_id) for user_id in user.sort_values('user_id')['user_id']
        ],
        dtype='int8')
