from collections import Counter
import pandas as pd
import itertools
from time import time


def get_data(direcname, tset):  # important: tset muss /set heissen
    all_data = pd.read_csv("dataset/" + direcname + tset, sep="#", index_col=False).T
    return dict(itertools.zip_longest(all_data.iloc[0], all_data.iloc[1].values))


class NearestNeighbours:

    def __init__(self, k):  # k is the number of neighbours which count
        self.k = k

    def levenshtein(self,s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[
                                 j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1  # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


    def find_neighbours2(self, word1, data):
        # dist = len(word1)
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
        #print(dists.values())
        return dists.values()

    def desicion(self,cities,data):
        countries = []
        for citygr in cities:
            for city in citygr:
                countries.append(data.get(city))
        frequency = Counter(countries)
        v = list(frequency.values())
        k = list(frequency.keys())
        return k[v.index(max(v))]


model = NearestNeighbours(9)    # 9: 0.76
train_data = get_data("ten_countries", "/train")
valid_data = get_data("ten_countries", "/valid")
neigh = model.find_neighbours2("Konstantinopel", train_data)
pred = model.desicion(neigh,train_data)
print(pred)
#print("got data")
time1 = time()
r = 0
y = list(valid_data.values())
"""for i,name in enumerate(valid_data.keys()):
    neigh = model.find_neighbours2(name, train_data)
    pred = model.desicion(neigh,train_data)
    gold = y[i]
    if pred == gold:
        try:
            print(r/i)
        except:
            pass
        r += 1

    else:
        print("failed" + name + " " + pred)

print(time()-time1)"""
