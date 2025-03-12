# Anyskin slip detection

This code reproduce the slip detection of the [anyskin project](https://any-skin.github.io/). The model is based on the [anyskin handoff demo](https://github.com/NYU-robot-learning/AnySkin-Handoff-Demo).

## Installation

After cloning the repo, you can simply use pip to install the dependancies. We recommend using a virtual environement.


```
pip install .
```

## Train

Dataset will be automatically downloaded from [HF hub](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection).

```python
python src/train.py
```

The best model is saved in the checkpoint folder, as well as its parameters.

## Test

```python
python src/pred.py
```
should output

```
Result : [[0.9996903]]
```


## Misc

The pt model can be converted in safetensors model using

```python
python src/convert.py
```
