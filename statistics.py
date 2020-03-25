from glob import glob
from os.path import join
from itertools import chain
from collections import Counter, defaultdict
from scipy.stats import pearsonr
from csv import reader

POEM_FOLDER = "poems"
REAL_POEMS_FOLDER = "real_poems"

def is_annotation_correct(left_real, right_real, score):
    if score.strip() == "1":
        return left_real and not right_real
    elif score.strip() == "2":
        return not left_real and right_real
    elif score.strip() == "Both":
        return left_real and right_real
    elif score.strip() == "None":
        return not left_real and not right_real
    else:
        raise(ValueError("Unknown label: " + score))

def get_real_poems():
    with open(glob(join(REAL_POEMS_FOLDER, "*"))[0]) as f:
        poems = f.read().split("\n\n")
        poems = ["\n".join(poem.split("\n")[:4]) if len(poem.split("\n")) > 4
                else poem for poem in poems]
        return poems

def get_real_percentage():
    real_poems = set()
    all_poems = set()
    correct = 0
    total = 0
    poems = get_real_poems()

    for filename in glob(join(POEM_FOLDER, "*.csv")):
        with open(filename, newline='') as f:
            lines = reader(f, dialect="unix")
            next(lines)
            for line in lines:
                if line[-1].strip():
                    if is_annotation_correct(line[0] in poems, line[1] in poems, line[-1]):
                        correct += 1
                        total += 1
                    else:
                        total += 1
                        
                for poem in line[0:2]:
                    if poem.replace(" \\\\ ", "\n") in poems:
                        real_poems.add(poem)
                    all_poems.add(poem)

    print(f"Real Poems in dataset: {100 * len(real_poems) / len(all_poems)}%, ({len(real_poems)})")
    print(f"Correct annotations for real poems: {100 * correct / total}%")


def compute_correlation():
    labels = defaultdict(list)
    for filename in glob(join(POEM_FOLDER, "*.csv")):
        with open(filename, newline='') as f:
            lines = reader(f, dialect="unix")
            categories = next(lines)[2:]

            for line in lines:
                pair = tuple(line[0:2])
                annotations = line[2:]
                if len([annotation for annotation in annotations if annotation.strip() != ""]) == 4:
                    for idx, annotation in enumerate(annotations):
                        label = None
                        if annotation.strip() == "1":
                            label = 1
                        elif annotation.strip() == "2":
                            label = -1
                        elif annotation.strip():
                            label = 0
                        else:
                            continue
                        labels[categories[idx]].append(label)

    pairs = list(labels.items())
    tuples = [(pair1, pair2) for pair1 in pairs for pair2 in pairs if pair1 != pair2]

    for (category1, labels1), (category2, labels2) in tuples:
        duplicate = (category2, labels2), (category1, labels1)
        if duplicate in tuples:
            tuples.remove(duplicate)

    for (category1, labels1), (category2, labels2) in tuples:
        print(f"* Pearson correlation between categories {category1} and {category2}: {pearsonr(labels1, labels2)[0]}")




def compute_statistics():
    unique_poems = set()
    unique_tuples = set()
    category_counter = defaultdict(Counter)

    for filename in glob(join(POEM_FOLDER, "*.csv")):
        with open(filename, newline='') as f:
            lines = reader(f, dialect="unix")
            categories = next(lines)[2:]

            for line in lines:
                pair = tuple(line[0:2])
                annotations = line[2:]
                unique_poems.update(pair)
                unique_tuples.add(pair)
                for idx, annotation in enumerate(annotations):
                    if annotation.strip():
                        category_counter[categories[idx]][pair] += 1

    return unique_poems, unique_tuples, category_counter

def print_statistics(unique_poems, unique_tuples, category_counter):

    total_annotations = 0
    total_annotations_per_count = defaultdict(int)
    for category, counter in category_counter.items():
        print(f"* Annotations for category {category}:")
        print(f"\t* contains {len(counter)} unique pairs")

        counts = defaultdict(list)
        for pair, count in counter.most_common():
            counts[count].append(pair)

        for count, pairs in counts.items():
            print(f"\t* {len(pairs)} pairs were annotated {count} times")
            total_annotations += count * len(pairs)
            total_annotations_per_count[count] += len(pairs)

    print(f"* {len(unique_poems)} unique poems in the dataset")
    print(f"* {len(unique_tuples)} unique pairs in the dataset")
    print(f"* {total_annotations} annotations in total in dataset")

    words = len(list(chain.from_iterable([poem.split() for poem in
                                          unique_poems])))

    print(f"* average word count of a poem is {words/len(unique_poems)}")

    for count, amount in total_annotations_per_count.items():
        print(f"* {amount} pairs were annotated {count} times")

    annotation_counter = Counter()
    for pair in unique_tuples:

        different_category_annotations = 0
        for category in category_counter.keys():
            if category_counter[category][pair] > 0:
                different_category_annotations += 1
        annotation_counter[different_category_annotations] += 1

    for idx in range(1, len(category_counter.keys())):
        print(f"* {annotation_counter[idx]} pairs were annotated in {idx} different categories")


if __name__ == "__main__":
    statistics = compute_statistics()
    print_statistics(*statistics)
    compute_correlation()
    get_real_percentage()
