import crowdgppl
import bws
from scipy.stats import spearmanr

def compute_correlation():
    bws_scores = bws.compute_scores()
    crowdgppl_scores = crowdgppl.compute_scores()
    return spearmanr(bws_scores, crowdgppl_scores)

if __name__ == "__main__":
    print(compute_correlation())
