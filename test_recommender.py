import numpy as np
import time
from datetime import timedelta
from app.recommender import Recommender

recommender = Recommender()

# def apk(actual, predicted, k=10):
#     if not actual:
#         return 0.0

#     if len(predicted)>k:
#         predicted = predicted[:k]

#     score = 0.0
#     num_hits = 0.0

#     for i,p in enumerate(predicted):
#         # first condition checks whether it is valid prediction
#         # second condition checks if prediction is not repeated
#         if p in actual and p not in predicted[:i]:
#             num_hits += 1.0
#             score += num_hits / (i+1.0)

#     return score / min(len(actual), k)

# def mapk(actual, predicted, k=10):
#     return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

# actual = list()
# predicted = list()
hit = {
    'with_model': 0,
    'no_model': 0,
}
confusion_matrix = {
    'with_model': np.array([[0,0],[0,0]]),
    'no_model': np.array([[0,0],[0,0]]),
}

users = recommender.matrix.index.tolist()
use_model = [False, True]
total_elapsed_time = 0
time_per_batch = 0
test_start = time.time()
for i, use in enumerate(use_model):
    key = 'with_model' if use else 'no_model'

    for index, user_id in enumerate(users):
        start = time.time()
        user_actual = list()
        user_predicted = list()
        _, _, result = recommender.get_anime_recommendation(
            user_id,
            similar_users_count=10,
            items=100,
            use_model=use,
            test_all_anime=False)

        # for res in result:
        #   if res == 'TP':
        #     user_actual.append(0)
        #     user_predicted.append(0)
        #   elif res == 'FP':
        #     user_actual.append(1)
        #     user_predicted.append(0)
        #   elif res == 'FN':
        #     user_actual.append(0)
        #     user_predicted.append(1)
        #   elif res == 'TN':
        #     user_actual.append(1)
        #     user_predicted.append(1)

        # actual.append(user_actual)
        # predicted.append(user_predicted)

        if result == 'TP':
            confusion_matrix[key][1][1] += 1
            hit[key] += 1.0
        elif result == 'FP':
            confusion_matrix[key][0][1] += 1
        elif result == 'FN':
            confusion_matrix[key][1][0] += 1
        elif result == 'TN':
            confusion_matrix[key][0][0] += 1
            hit[key] += 1.0

        end = time.time()

        prediction_time = end - start
        total_elapsed_time += prediction_time
        time_per_batch = ((time_per_batch + prediction_time) /
                          2) if time_per_batch > 0 else prediction_time
        remaining_time = time_per_batch * (len(users) - (index + 1))

        print("====================================================")
        print("Batch {}/{}".format(index + 1, len(users)))
        print("Total elapsed time: {}".format(
            timedelta(seconds=total_elapsed_time)))
        print("Estimated time remaining: {}".format(
            timedelta(seconds=remaining_time)))
        print("====================================================")

test_end = time.time()

print("Testing Time", timedelta(seconds=(test_end - test_start)))
print("Hit rate (with model): {:.3f}".format(hit["with_model"] / len(users)))
print("Hit rate (without model): {:.3f}".format(hit["no_model"] / len(users)))
print("Confusion Matrix (with model):")
print(confusion_matrix['with_model'])
print("Confusion Matrix (without model):")
print(confusion_matrix['no_model'])
