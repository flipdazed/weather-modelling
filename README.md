Deep Networks for Prediction of Weather Events (under dev)
---

This repository contains some of the code used for a research project at Auckland University. The code is based in `python-theano` and can consequently be shared however the underlying data sources remain hidden for confidentiality.

# Running Code
Code should always be run from the root directory for example

    ipython test/test_load.py

running with `python` may not pick up the directory structure correctly on some distributions like `Ubuntu` but `ipython` works across the board.

# Structure

 - `models` : building blocks to build the models
 - `utils` : non-core functions unrelated to ML such as `logging`
 - `dump` : stored data such as plots in `dump/plots/` and trained models in `dump/models`
 - `runlog.log` : will appear in the root directory. View with `cat runlog.log` for colour support
 - `data.py` : routines related to loading / storing data. Parses `config.ini`.
 - `config.ini` : configuration file described below

# Configuration

A `config.ini` file is required in the root directory that will be structured in standard `.ini` format as

```
[Global]
log = /path/to/root/runlog.log

[Load Data]
mnist_loc = /a/directory/path/mnist.pkl.gz
mnist_url = http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
icestorm_loc = /path/to/data/files/
model_dir = /path/to/root/dump/models/
```

and will be parsed by `py` where parameters are stored.