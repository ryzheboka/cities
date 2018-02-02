from collections import Counter
import pandas as pd
import itertools
from time import time


def get_data(dirname, tset):
    """

    :param dirname: name of directory with data
    :param tset: name of the file where to get data from
    :return: dictionary with city name as key and country as value
    """
    all_data = pd.read_csv("dataset/" + dirname + tset, sep="#", index_col=False).T
    return dict(itertools.zip_longest(all_data.iloc[0], all_data.iloc[1].values))


class NearestNeighbours:
    """
    functions for computing city problem with KNN
    """

    def __init__(self, k):
        """
        :param k: the number of nearest neighbours which count
        """
        self.k = k

    def levenshtein(self, s1, s2):
        """
        here distance function
        :param s1: first word
        :param s2: second word
        :return: minimal number of changes to make the first word to the second word
        """
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[
                                 j + 1] + 1
                deletions = current_row[j] + 1  # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def find_neighbours(self, word1, data):
        """

        :param word1: city name to predict
        :param data: dictionary with city name as key and country as value
        :return: k nearest neighbours of the given city
        """
        dists = {}
        for word2 in list(data.keys()):
            new = self.levenshtein(word1, word2)
            if sum(len(v) for v in dists.values()) < self.k:
                if new in dists.keys():
                    dists[new].append(word2)
                else:
                    dists[new] = [word2]
            else:
                m = max(dists)
                if m > new:
                    dists.pop(m)
                    if new in dists.keys():
                        dists[new].append(word2)
                    else:
                        dists[new] = [word2]
        return dists.values()

    @staticmethod
    def decision(cities, data):
        """
        can be optimized: unnecessary to search for the keys of neighbours again, it would be
        better to return them in "find_neighbours"
        :param cities: k nearest neighbours of the city to predict
        :param data: dictionary with city name as key and country as value
        :return: the prediction (a country)
        """
        countries = []
        for citygr in cities:
            for city_name in citygr:
                countries.append(data.get(city_name))
        frequency = Counter(countries)
        v = list(frequency.values())
        k = list(frequency.keys())
        return k[v.index(max(v))]


model = NearestNeighbours(9)
train_data = get_data("ten_countries", "/train")
print(train_data)
valid_data = get_data("ten_countries", "/valid")

want_to_stay = True
while want_to_stay:
    city = input("City to predict:  ")
    if city == "STOP":
        want_to_stay = False
    else:
        neigh = model.find_neighbours(city, train_data)
        out = model.decision(neigh, train_data)
        print(out)

time1 = time()
r = 0
y = list(valid_data.values())

# should add computing accuracy (not on training data)
