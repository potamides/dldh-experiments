from glob import glob
from os import getcwd
from os.path import join
from collections import Counter, OrderedDict
from csv import reader

POEM_FOLDER = "poems"

def load_dataset():
    most_positive = OrderedDict()
    most_negative = OrderedDict()
    total = OrderedDict()

    for filename in glob(join(POEM_FOLDER, "*.csv")):
        with open(filename, newline='') as f:
            lines = reader(f, dialect="unix")
            next(lines)
            for line in lines:
                left_poem, right_poem = line[0:2]
                labels = line[2:]
                for poem in [left_poem, right_poem]:
                    for counter in [most_positive, most_negative, total]:
                        if poem not in counter:
                            counter[poem] = 0
                for label in labels:
                    if label.strip() == "1":
                        most_positive[left_poem] += 1
                        most_negative[right_poem] += 1
                    elif label.strip() == "2":
                        most_positive[right_poem] += 1
                        most_negative[left_poem] += 1

                    if label.strip():
                        total[right_poem] += 1
                        total[left_poem] += 1


    return most_positive, most_negative, total

def compute_scores():
    most_positive, most_negative, total = load_dataset()
    scores = list()
    for poem in most_positive.keys():
        scores.append((most_positive[poem] - most_negative[poem]) /
                      max((total[poem], 1)))
    return scores


if __name__ == "__main__":
    print(compute_scores())
