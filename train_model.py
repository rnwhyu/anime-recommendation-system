import math
import pickle
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

DATA_DIR = "./data"

ANIME_CLEANED = "{}/anime_cleaned.csv".format(DATA_DIR)
USER_CLEANED = "{}/users_cleaned_1000.csv".format(DATA_DIR)
PREPROCESSED_INPUT = "{}/preprocessed_input_1000.csv".format(DATA_DIR)


def list_chunk(list, chunk_size):
    for i in range(0, len(list), chunk_size):
        yield list[i:i + chunk_size]


def batch_predict(model, X, num_of_batches):
    total_elapsed_time = 0
    time_per_batch = 0
    y_pred = None

    print("====================================================")
    print("Predicting in batches")
    print("====================================================")
    for index, batch in enumerate(X):
        start = time.time()
        y_pred = model.predict(batch) if y_pred is None else np.concatenate(
            (y_pred, model.predict(batch)))
        end = time.time()

        prediction_time = end - start
        total_elapsed_time += prediction_time
        time_per_batch = ((time_per_batch + prediction_time) /
                          2) if time_per_batch > 0 else prediction_time
        remaining_time = time_per_batch * (num_of_batches - (index + 1))

        print("====================================================")
        print("Batch {}/{}".format(index + 1, num_of_batches))
        print("Total elapsed time: {}".format(
            timedelta(seconds=total_elapsed_time)))
        print("Estimated time remaining: {}".format(
            timedelta(seconds=remaining_time)))
        print("====================================================")

    print("====================================================")
    print("Finished predicting")
    print("====================================================")

    return y_pred


def batch_predict_proba(model, X, num_of_batches):
    total_elapsed_time = 0
    time_per_batch = 0
    y_pred = None

    print("====================================================")
    print("Predicting in batches")
    print("====================================================")
    for index, batch in enumerate(X):
        start = time.time()
        pred = model.predict_proba(batch)
        pred = ['Suka' if pr[0] > pr[1] else 'Tidak Suka' for pr in pred]
        y_pred = pred if y_pred is None else np.concatenate((y_pred, pred))
        end = time.time()

        prediction_time = end - start
        total_elapsed_time += prediction_time
        time_per_batch = ((time_per_batch + prediction_time) /
                          2) if time_per_batch > 0 else prediction_time
        remaining_time = time_per_batch * (num_of_batches - (index + 1))

        print("====================================================")
        print("Batch {}/{}".format(index + 1, num_of_batches))
        print("Total elapsed time: {}".format(
            timedelta(seconds=total_elapsed_time)))
        print("Estimated time remaining: {}".format(
            timedelta(seconds=remaining_time)))
        print("====================================================")

    print("====================================================")
    print("Finished predicting")
    print("====================================================")

    return y_pred


user_cols = ['username', 'user_id']
user = pd.read_csv(USER_CLEANED, usecols=user_cols)
print(user.shape, '\n', user.head())

print('\n\n\n')

anime_cols = [
    'anime_id', 'title', 'title_english', 'score', 'scored_by', 'rank',
    'popularity', 'members', 'favorites', 'genre'
]
anime = pd.read_csv(ANIME_CLEANED, usecols=anime_cols)
print(anime.shape, '\n', anime.head())

anime_genres = anime.filter('genre')
anime_genres['genre'] = anime['genre'].str.split(',')
list_of_genre = anime_genres.explode('genre')
list_of_genre = list_of_genre.drop_duplicates('genre').filter(['genre'])
list_of_genre = list_of_genre.dropna()
list_of_genre = [genre.strip() for genre in list_of_genre['genre'].unique()]
list_of_genre = list(set(list_of_genre))

print(anime.info(verbose=False, memory_usage='deep'), '\n')
print(user.info(verbose=False, memory_usage='deep'), '\n')

# run this block to import the exported preprocessed_input.csv
preprocessed_input_dtype = {"my_score": "int8", "label": "category"}
for genre in list_of_genre:
    preprocessed_input_dtype[genre] = "int8"

for user_id in user['user_id']:
    preprocessed_input_dtype[str(user_id)] = "int8"

for anime_id in anime['anime_id']:
    preprocessed_input_dtype["a_{}".format(str(anime_id))] = "int8"

preprocessed_input = pd.read_csv(
    PREPROCESSED_INPUT, dtype=preprocessed_input_dtype).reset_index(drop=True)

# random under sampling
# sampler = RandomUnderSampler(positive_indicator='Suka',
#                              negative_indicator='Tidak Suka')
# preprocessed_input = sampler.transform(preprocessed_input)

print(preprocessed_input.shape)
print(preprocessed_input.head())

print(preprocessed_input.info(verbose=False, memory_usage="deep"))

# model input
X = preprocessed_input[preprocessed_input.columns[:-1]]
X = X.drop(columns=['my_score'])

# model label
y = preprocessed_input['label']

# scaler = MinMaxScaler()
# scaler.fit_transform(X)
# X = scaler.transform(X)
# print(X)

# filename = 'scaler.pkl'
# path = './export/' + filename

# with open(path, 'wb') as model_file:
#     pickle.dump(scaler, model_file)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=109)  # 80% training and 20% test
print(np.unique(y_train, return_counts=True))

n_estimators = math.ceil(X_train.shape[0] / 10000)

start = time.time()

# svc = svm.SVC(C=1.0, kernel='poly', gamma='scale')
svc = svm.LinearSVC(dual=False, class_weight=None, C=3.5)
clf = BaggingClassifier(
    base_estimator=svc,
    n_jobs=-1,  # turn this off if too much resources are used
    verbose=3,
    max_samples=1.0 / n_estimators,
    n_estimators=n_estimators)

clf.fit(X_train, y_train)

end = time.time()
print("Training Time", timedelta(seconds=(end - start)))

filename = 'classifier_model.pkl'
path = './export/' + filename

with open(path, 'wb') as model_file:
    pickle.dump(clf, model_file)

# Predict the response for test dataset
X_test_batches = list_chunk(X_test, 1000)
y_pred = batch_predict(clf, X_test_batches, len(list(list_chunk(X_test,
                                                                1000))))
# y_pred = clf.predict(X_test)
print(np.unique(y_pred, return_counts=True))

# Import scikit-learn metrics module for accuracy calculation

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred, pos_label="Suka"))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred, pos_label="Suka"))

print(metrics.confusion_matrix(y_test, y_pred, labels=['Tidak Suka', 'Suka']))
print(
    '(tn, fp, fn, tp)',
    metrics.confusion_matrix(y_test, y_pred, labels=['Tidak Suka',
                                                     'Suka']).ravel())

trimmed_input = preprocessed_input.sample(10000)
# model input
X_test = trimmed_input[trimmed_input.columns[:-1]]
X_test = X_test.drop(columns=['my_score'])

# model label
y_test = trimmed_input['label']

# X_test = scaler.transform(X_test)

# Predict the response for trimmed dataset
y_pred = clf.predict(X_test)
print(np.unique(y_pred, return_counts=True))

# Import scikit-learn metrics module for accuracy calculation

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred, pos_label="Suka"))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred, pos_label="Suka"))

print(metrics.confusion_matrix(y_test, y_pred, labels=['Tidak Suka', 'Suka']))
print(
    '(tn, fp, fn, tp)',
    metrics.confusion_matrix(y_test, y_pred, labels=['Tidak Suka',
                                                     'Suka']).ravel())

# Predict the response for trimmed dataset
y_pred = clf.predict(X_test)

print(np.unique(y_pred, return_counts=True))

# Import scikit-learn metrics module for accuracy calculation

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred, pos_label="Suka"))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred, pos_label="Suka"))

print(metrics.confusion_matrix(y_test, y_pred, labels=['Tidak Suka', 'Suka']))
print(
    '(tn, fp, fn, tp)',
    metrics.confusion_matrix(y_test, y_pred, labels=['Tidak Suka',
                                                     'Suka']).ravel())
