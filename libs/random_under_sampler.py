import numpy as np


class RandomUnderSampler:

    def __init__(self,
                 label_column='label',
                 positive_indicator=1,
                 negative_indicator=0,
                 positive_size=0.5,
                 negative_size=0.5):
        self.label_column = label_column
        self.positive_indicator = positive_indicator
        self.negative_indicator = negative_indicator
        self.positive_size = positive_size
        self.negative_size = negative_size

    def transform(self, data):
        df = data.copy()
        positive_count = 0
        negative_count = 0

        (indicators, counts) = np.unique(data[self.label_column],
                                         return_counts=True)
        if self.positive_indicator == indicators[1]:
            self.positive_indicator = indicators[1]
            self.negative_indicator = indicators[0]

            positive_count = counts[1]
            negative_count = counts[0]
        else:
            positive_count = counts[0]
            negative_count = counts[1]

        positive_indicies = data[data[self.label_column] ==
                                 self.positive_indicator].index
        negative_indicies = data[data[self.label_column] ==
                                 self.negative_indicator].index

        if self.positive_size == self.negative_size:
            if positive_count > negative_count:
                random_indices = np.random.choice(positive_indicies,
                                                  negative_count,
                                                  replace=False)
                indices = np.concatenate([negative_indicies, random_indices])
                df = df.loc[indices].reset_index(drop=True)
            else:
                random_indices = np.random.choice(negative_indicies,
                                                  positive_count,
                                                  replace=False)
                indices = np.concatenate([positive_indicies, random_indices])
                df = df.loc[indices].reset_index(drop=True)
        else:
            if self.positive_size > self.negative_size:
                positive_indicies = np.random.choice(positive_indicies,
                                                     int(self.positive_size *
                                                         positive_count),
                                                     replace=False)

                negative_indicies = np.random.choice(negative_indicies,
                                                     int(self.negative_size *
                                                         negative_count),
                                                     replace=False)

                indices = np.concatenate(
                    [positive_indicies, negative_indicies])
                df = df.loc[indices].reset_index(drop=True)
            else:
                positive_indicies = np.random.choice(positive_indicies,
                                                     int(self.positive_size *
                                                         positive_count),
                                                     replace=False)

                negative_indicies = np.random.choice(negative_indicies,
                                                     int(self.negative_size *
                                                         negative_count),
                                                     replace=False)

                indices = np.concatenate(
                    [positive_indicies, negative_indicies])
                df = df.loc[indices].reset_index(drop=True)

        df = df.sample(frac=1).reset_index(drop=True)

        return df
