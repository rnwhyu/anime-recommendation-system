import pandas as pd
from pandas.api.types import CategoricalDtype

pd.options.mode.chained_assignment = None

ANIME_CLEANED_CSV = "./data/anime_cleaned.csv"
PREPROCESSED_ANIME_CSV = "./data/preprocessed_anime.csv"

anime = pd.read_csv(ANIME_CLEANED_CSV,
                    usecols=[
                        'anime_id', 'title', 'title_english', 'image_url',
                        'score', 'scored_by', 'rank', 'popularity', 'members',
                        'favorites', 'genre'
                    ]).dropna()

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

preprocessed_anime = pd.read_csv(PREPROCESSED_ANIME_CSV,
                                 dtype=preprocessed_anime_dtype)


def get_all_anime():
    return anime


def filter_anime_by_indices(indices):
    df = anime[anime['anime_id'].isin(indices)]
    df['image_url'] = df['image_url'].str.replace(
        'https://myanimelist.cdn-dena.com/',
        'https://cdn.myanimelist.net/',
        regex=False)

    categorical_indices_order = CategoricalDtype(indices, ordered=True)
    df['anime_id'] = df['anime_id'].astype(categorical_indices_order)
    df = df.sort_values(['anime_id'])
    return df


def get_all_anime_genre():
    return list_of_genre


def get_all_preprocessed_anime():
    return preprocessed_anime


def get_all_anime_id():
    return [str(anime_id) for anime_id in anime['anime_id']]
