import pandas as pd
import itertools
import random
import copy


def get_data(dirname, tset):
    """

       :param dirname: name of directory with data
       :param tset: name of the file where to get data from
       :return: dictionary with city name as key and country as value
       """
    all_data = pd.read_csv("dataset/" + dirname + tset, sep="#", index_col=False).T
    return dict(itertools.zip_longest(all_data.iloc[0], all_data.iloc[1].values))


class NearestNeighbours:
    def __init__(self, k0):  # k is the number of neighbours which count
        """
        :param k0: the number of nearest neighbours which count
        """
        self.k = k0

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
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def change_letter(word, chars):
        """
        :param word: word to be changed
        :param chars: chars that are probably in a similar name
        :return: list of all possible changes to the word
         (removing one letter, adding one letter, replacing one letter)
        """
        word = list(word)
        changed = list()
        for i, l in enumerate(word):
            word_remove = copy.copy(word)
            del word_remove[i]
            changed.append("".join(word_remove))
            for char in chars:
                word_add = copy.copy(word)
                word_add.insert(i, char)
                changed.append("".join(word_add))
                word_replace = copy.copy(word)
                word_replace[i] = char
                changed.append("".join(word_replace))
        return changed

    @staticmethod
    def capital_and_lowercase(word):
        """

        :param word: city name
        :return: same city name with correct lower and uppercase letters
        """
        word = word.lower()
        return "".join(
            c.upper() if word[i - 1] == " " or word[i - 1] == "-" or i == 0 else c for i, c in enumerate(word))

    def prove_changed_versions(self, changed, rand_set):
        """
        :param changed: changed versions of one name
        :param rand_set: random cities of the same country
        :return: dictionary with changed names as keys and sum
        of their levenshtein-distances to names in rand_set as values
        """
        w_ranks = {k: 0 for k in set(changed)}

        for wordVersion in set(changed):
            for city in rand_set:
                lev = self.levenshtein(city, wordVersion)
                if lev == 0:
                    w_ranks.pop(wordVersion)
                else:
                    if wordVersion in w_ranks.keys():
                        w_ranks[wordVersion] += lev
        return w_ranks

    def create(self, cntr, data):
        """

        :param cntr: country for an imaginary name
        :param data: dictionary with city name as key and country as value
        :return: imaginary city name
        """
        country_indeces = [i for i, x in enumerate(list(data.values())) if x == cntr]
        country_names = [list(data.keys())[i] for i in country_indeces]
        n = random.randint(1, len(country_names))

        rand_city = country_names[n]
        rand_set = self.find_neighbours(rand_city, country_names)
        rand_set = [char for itr_rand_set in rand_set for char in list(itr_rand_set)]
        used_chars = [char for itrRandSet in rand_set for chars in list(itrRandSet) for char in list(chars)]
        used_chars = set(used_chars)

        changed_versions = self.change_letter(word=rand_city, chars=used_chars)

        for city in rand_set:
            changed_versions += (self.change_letter(word=city, chars=used_chars))

        w_ranks = self.prove_changed_versions(changed_versions, rand_set)
        answer = min(w_ranks, key=w_ranks.get)
        return self.capital_and_lowercase(answer)

    def find_neighbours(self, word1, data):
        """

        :param word1: city name to predict
        :param data: dictionary with city name as key and country as value
        :return: k nearest neighbours of the given city
        """
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
model = NearestNeighbours(kmod)
train_data = get_data("ten_countries", "/train")
valid_data = get_data("ten_countries", "/valid")
print(set(train_data.values()))

wantToStay = True
while wantToStay:
    country = input("Country:   ")
    if country in set(train_data.values()):
        print(model.create(country, train_data))
    elif country == "STOP":
        userWantsToStay = False
    else:
        print("This country is not in the data")
