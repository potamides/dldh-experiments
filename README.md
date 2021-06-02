# Best-Worst Counting and Gaussian Process Preference Learning: A Detailed Comparison

Code for the experiments conducted in the context of the [Deep Learning & Digital Humanities Seminar](https://github.com/SteffenEger/dldh).
For a detailed analysis of the results please take a look at this [term paper](https://drive.google.com/uc?print=false&id=1QNawSRoNC_pZG5yC9Vij5n08mwdKdwiD).

## Setup

Clone the repository:
```sh
git clone --recurse-submodules git@github.com:DrCracket/dldh-experiments.git
```

Install requirements:

```sh
pip install -r requirements.txt
```

Put the annotated poetry data in to the `poems` folder. The files are expected to be in unix-flavored csv format.
Put the file with real poems in the `real_poems` folder. The file should
contain all real poems separated with a newline.

## Experiments

Train models and compute features:
```sh
python gppl.py && python crowdgppl.py
```

Run the experiments:
```sh
python main.py
```

Compute various statistics of the dataset:
```sh
python statistics.py
```
