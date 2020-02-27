from glob import glob
from os.path import join
from itertools import chain
from collections import Counter, defaultdict
from scipy.stats import pearsonr

POEM_FOLDER = "poems"

def compute_correlation():
    labels = defaultdict(list)
    with open(join(POEM_FOLDER, "poems_200 - poems_200.tsv")) as f:
        categories = f.readline().strip("\n").split("\t")[2:]

        for line in f.readlines():
            line = line.split("\t")
            pair = tuple(line[0:2])
            annotations = line[2:]
            for idx, annotation in enumerate(annotations):
                label = None
                if annotation.strip() == "1":
                    label = 1
                elif annotation.strip() == "2":
                    label = 2
                else:
                    label = 0
                labels[categories[idx]].append(label)

    pairs = list(labels.items())
    tuples = [(pair1, pair2) for pair1 in pairs for pair2 in pairs if pair1 != pair2]

    for (category1, labels1), (category2, labels2) in tuples:
        print(f"* Pearson correlation between categories {category1} and {category2}: {pearsonr(labels1, labels2)[0]}")




def compute_statistics():
    unique_poems = set()
    unique_tuples = set()
    category_counter = defaultdict(Counter)

    for filename in glob(join(POEM_FOLDER, "*.tsv")):
        print(filename)
        with open(filename) as f:
            categories = f.readline().strip("\n").split("\t")[2:]

            for line in f.readlines():
                line = line.split("\t")
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
