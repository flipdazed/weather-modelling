Deep Networks for Prediction of Weather Events 
---

This repository contains some of the code used for a research project at Auckland University. The code is based in `python-theano` and can consequently be shared however the underlying data sources remain hidden for confidentiality.\

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

and will be parsed by `load.py` where parameters are stored.