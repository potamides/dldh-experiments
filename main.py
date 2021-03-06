import gppl
import crowdgppl
import bws
from collections import OrderedDict
from statistics import get_real_poems
from scipy.stats import spearmanr, linregress
from sys import path
import matplotlib.pyplot as plt
import numpy as np

def get_best_poems():
    #print_this = OrderedDict()
    for name, model in (("crowdGPPL", crowdgppl), ("GPPL", gppl), ("BWS", bws)):
        poems = tuple(model.load_dataset())[-1]
        scores = np.squeeze(model.compute_scores())
        both = sorted(zip(scores, poems), reverse=True, key=lambda x: x[0])
        real_poems = get_real_poems()
        real_best100 = len([poem for (score, poem) in both[:100] if poem in real_poems])
        real_worst100 = len([poem for (score, poem) in both[-100:] if poem in real_poems])
        #for idx, (score, poem) in enumerate(both, 1):
        #    if poem in print_this:
        #        print_this[poem] += f";{score}"
        #    else:
        #        print_this[poem] = f"Stanza_{idx};{score}"
        #        #print(idx)
        #        #print(poem + "\n")

        print(f"For {name} real poems under the best 100: {real_best100}%")
        print(f"For {name} real poems under the worst 100: {real_worst100}%")

        print(f"\n## Best {name} poems:")
        for score, poem in  both[:5]:
            print(f"\n### Score: {score}, is real: {poem in real_poems}\n")
            print(poem)
        print(f"\n## Worst {name} poems:")
        for score, poem in  both[-5:]:
            print(f"\n### Score: {score}, is real: {poem in real_poems}\n")
            print(poem)

    #for poem in print_this.keys():
    #    print_this[poem] += f";{poem in real_poems}"

    #print("Stanza_{idx};{crowdGPPL_score};{GPPL_score};{BWS_score};{real}")
    #print("\n".join(list(print_this.values())))

def linear_regression(x, y):
    gradient, intercept, r_value, p_value, std_err = linregress(x,y)
    mn=np.min(x)
    mx=np.max(x)
    x1=np.linspace(mn,mx,500)
    y1=gradient*x1+intercept

    return x1, y1

def plot_gppl():
    crowdgppl_scores = np.squeeze(crowdgppl.compute_scores())
    gppl_scores = np.squeeze(gppl.compute_scores())
    print(f"Range of GPPL scores: [{np.mean(gppl_scores[gppl_scores < 0])}, {np.mean(gppl_scores[gppl_scores >= 0])}]")
    print(f"Range of crowdGPPL scores: [{np.mean(crowdgppl_scores[crowdgppl_scores < 0])}, {np.mean(crowdgppl_scores[crowdgppl_scores >= 0])}]")
    plt.scatter(crowdgppl_scores, gppl_scores)
    plt.plot(*linear_regression(crowdgppl_scores, gppl_scores), c="red")
    plt.xlabel("crowdGPPL scores")
    plt.ylabel("GPPL scores")
    plt.show()

def plot_lengthscales():
    scales = gppl.get_lengthscales()
    if isinstance(scales, int):
        return
    embedding_scales = np.sort(scales[0:-3])
    manual_feature_scales = scales[-3:]
    positions = np.searchsorted(embedding_scales, manual_feature_scales)
    plt.plot(embedding_scales, label=f"Sorted {len(embedding_scales)}-dimensional embedding length-scales")
    plt.scatter(positions[0], manual_feature_scales[0],
                label=f"Rhyme length-scale (index={positions[0]})", zorder=100)
    plt.scatter(positions[1], manual_feature_scales[1],
                label=f"Alliteration length-scale (index={positions[1]})", zorder=100)
    plt.scatter(positions[2], manual_feature_scales[2],
                label=f"Readability length-scale (index={positions[2]})", zorder=100)
    plt.legend()
    plt.show()

def compute_correlation():
    bws_scores = "BWS", bws.compute_scores()
    crowdgppl_scores = "crowdGPPL", crowdgppl.compute_scores()
    gppl_scores = "GPPL", gppl.compute_scores()
    results = sorted([bws_scores, gppl_scores, crowdgppl_scores])
    tuples = [sorted((pair1, pair2)) for pair1 in results for pair2 in results
              if pair1 != pair2 and pair1[0] < pair2[0]]
    for (type1, scores1), (type2, scores2) in tuples:
        correlation = spearmanr(scores1, scores2)
        print(f"Correlation between {type1} and {type2}: {correlation[0]}")

if __name__ == "__main__":
    get_best_poems()
    compute_correlation()
    plot_gppl()
    plot_lengthscales()
