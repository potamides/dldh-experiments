import gppl
import crowdgppl
import bws
from scipy.stats import spearmanr
from sys import path


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
    compute_correlation()
