# ANIME RECOMMENDER

This recommender is developed using user-based collaborative filtering and SVM classifier. The data is acquired from [kaggle](https://www.kaggle.com/datasets/azathoth42/myanimelist).

The model is trained by using `1000 user` at max, because hardware limitation.  Even if it's only `1000 user`, the preprocessed input's dimension are  somewhere around `390785x3534`, which is very huge considering the fact that the exported CSV size is reaching `±5 GB` in total.

The way this recommendation system works is by using cosine similarity to find `k` similar user and decide the top `n` anime based on each similar user scoring. The SVM classifier predict the selected user's disliked animes based on each anime's genre and then remove it from the recommendation list.

## Setup & Installation

1. Clone this repo & move to its directory
2. Activate your virtual env
3. Install the required packages with `pip install -r requirements.txt`
4. And you're good to go

## How To Get Recommendation

### Webserver
1. Copy or rename `.env.example` to `.env` and set it up accodingly
2. Run `flask run` in terminal
3. Open the provided localhost url
4. Select user and wait for their result

### CLI
1. Follow instructions in [DATA](./data/README.md) directory
2. Follow instructions in [EXPORT](./export/README.md) directory
3. Run `python recommender.py`
4. And the top 5 anime recommendation for user with the username [`Zexu`](https://myanimelist.net/profile/Zexu) or user_id `459521` will be shown by default (might want to edit this in the file directly because it will cause error if the user doesn't exist)