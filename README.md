# dldh-experiments

Code for the experiments conducted in the context of the [Deep Learning & Digital Humanities Seminar](https://github.com/SteffenEger/dldh).
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

An overview of the results can be found [here](https://hackmd.io/TqQqj5r1QoS-MoTSbYpQ5g).
