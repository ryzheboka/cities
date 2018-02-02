import pandas as pd
import itertools
import random
import copy


def get_data(direcname, tset):  # important: tset muss /set heissen
    all_data = pd.read_csv("dataset/" + direcname + tset, sep="#", index_col=False).T
    return dict(itertools.zip_longest(all_data.iloc[0], all_data.iloc[1].values))


class NearestNeighbours:

    def __init__(self, k0):  # k is the number of neighbours which count
        self.k = k0

    def levenshtein(self, s1, s2):
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
                                 j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


    def changeLetter(self, word, chars):
        word = list(word)
        changed = list()
        for i, l in enumerate(word):
            wordRemove = copy.copy(word)
            del wordRemove[i]
            changed.append("".join(wordRemove))
            for char in chars:
                wordAdd = copy.copy(word)
                wordAdd.insert(i,char)
                changed.append("".join(wordAdd))
                wordReplace = copy.copy(word)
                wordReplace[i] = char
                changed.append("".join(wordReplace))
        return changed

    def capitalAndLowercase(self, word):
        word = word.lower()
        return "".join(c.upper() if word[i-1] == " " or word[i-1] == "-" or i == 0 else c for i, c in enumerate(word))


    def proveChangedVersions(self, changed, randSet):
        wVRanks = {k: 0 for k in set(changed)}

        for wordVersion in set(changed):
            #   print(wordVersion)
            for city in randSet:
                lev = self.levenshtein(city, wordVersion)
                if lev == 0:
                    #   print("Warning")
                    wVRanks.pop(wordVersion)
                else:
                    if wordVersion in wVRanks.keys():
                        wVRanks[wordVersion] += lev
        return wVRanks

    def create(self, country, data):
        countryIndeces = [i for i, x in enumerate(list(data.values())) if x == country]
        countryNames = [list(data.keys())[i] for i in countryIndeces]
        n = random.randint(1,len(countryNames))

        randCity = countryNames[n]
        #   print(randCity)
        randSet = self.find_neighbours2(randCity,countryNames)
        randSet = [char for itrRandSet in randSet for char in list(itrRandSet)]
        usedChars = [char for itrRandSet in randSet for chars in list(itrRandSet) for char in list(chars)]
        usedChars = set(usedChars)

        changedVersions = self.changeLetter(word=randCity,chars=usedChars)
        #   print(randSet)


        for city in randSet:
            #   print("city "+city)
            changedVersions += (self.changeLetter(word=city, chars=usedChars))

        wVRanks = self.proveChangedVersions(changedVersions, randSet)
        answer = min(wVRanks, key=wVRanks.get)
        return self.capitalAndLowercase(answer)


    def find_neighbours2(self, word1, data):
        # dist = len(word1)
        dists = {}
        for word2 in data:
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


kmod = 4
model = NearestNeighbours(kmod)    # 9: 0.76
train_data = get_data("ten_countries", "/train")
valid_data = get_data("ten_countries", "/valid")
print(set(train_data.values()))
t = True
while(t):
    country = input("Country:   ")
    if country in set(train_data.values()):
        print(model.create(country, train_data))
    elif country == "STOP":
        t = False
    else:
        print("This country is not in the data")
