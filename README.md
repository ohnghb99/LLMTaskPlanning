# Benchmarking LLM-based Task Planners

## Environment

Ubuntu 14.04+ is required. The scripts were developed and tested on Ubuntu 22.04 and Python 3.8.

You can use WSL-Ubuntu on Windows 10/11.

## Install

We recommends using a virtual environment.

```bash
$ conda create -n {env_name} python=3.8
$ conda activate {env_name}
```
Install PyTorch (>=1.11.0) first (see https://pytorch.org/get-started/locally/),
then install python packages in `requirements.txt`.

```bash
$ pip install -r requirements.txt
```

Download ALFRED dataset.

```bash
$ cd alfred/data
$ sh download_data.sh json
```


## Benchmarking on ALFRED

```bash
$ python src/evaluate.py --config-name='config_alfred'
```

You can override the configuration. We used [Hydra](https://hydra.cc/) for configuration management.

```bash
$ python evaluate.py --config-name='config_alfred' planner.model='EleutherAI/gpt-neo-125M'
```


## Benchmarking on Watch-And-Help
```bash
$ cd {project_root}
$ ./script/icra_exp1_benchmark_wah.sh
```
