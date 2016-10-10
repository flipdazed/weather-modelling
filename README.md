Deep Networks for Prediction of Weather Events
===

**Status** : under development

This repository contains some of the code used for a research project at Auckland University. The code is based in `python-theano` and can consequently be shared however the underlying data sources remain hidden for confidentiality.

## Table of Contents
 - [Running Code](#running)
   - [Run Logs](#running-logs)
   - [Avoiding System Sleeping](#running-sleeping)
 - [Structure](#structure)
 - [Configuration](#config)

<a name="running"/>
# Running Code
Code should always be run from the root directory for example

    ipython test/test_mlp.py

![](/../screenshots/screenshots/runtime_screenshot.png "a runtime example")

running with `python` may not pick up the directory structure correctly on some distributions like `Ubuntu` but `ipython` works across the board. 

<a name="running-logs"/>
## Run Logs
Logs are stored by overwriting the default file `runlog.log` which will contain `OS X` terminal colour codes to highlight different information levels. The log should therefore be opened in the terminal though a command such as `cat`

![](/../screenshots/screenshots/log_screenshot.png "a runlog.log example")

<a name="running-sleeping"/>
## Avoiding System Sleeping
All scripts contain [`import caffeine`](https://pypi.python.org/pypi/caffeine/0.2) which prevents the system from going into sleep during runtime.

<a name="structure"/>
# Structure

 - `train_[MODEL].py` : trains `MODEL` on the weather data
 - `predict.py` : make a prediction from data by loading a trained model
 - `models` : building blocks to build the models
 - `test` : contains test routines for each model in `models` using MNIST
 - `utils` : non-core functions unrelated to ML such as `logging`
 - `dump` : stored data such as plots in `dump/plots/` and trained models in `dump/models`
 - `runlog.log` : will appear in the root directory. View with `cat runlog.log` for colour support
 - `data.py` : routines related to loading / storing data. Parses `config.ini`.
 - `config.ini` : configuration file described below

<a name="configuration"/>
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