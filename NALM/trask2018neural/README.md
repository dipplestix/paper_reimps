# Trask et al. (2018) – Extrapolation Failures in Neural Networks

This directory contains a compact reproduction of the failure mode highlighted in
Figure 1 of *Numerical Extrapolation Failures in Neural Networks* (Trask et al., 2018).
The experiment trains a narrow multilayer perceptron (MLP) to copy its scalar input on
a bounded interval.  Although the model can fit the training data perfectly, it fails to
extrapolate when queried with larger magnitude values.

## Usage

```bash
pip install -r requirements.txt  # optional if the repository-wide environment lacks torch/matplotlib
python -m trask2018neural.identity_failure
```

Running the script produces a figure at
`trask2018neural/outputs/identity_failure.png` that mimics the qualitative behaviour
reported in the paper: the training loss falls close to zero, but the predictions
outside the training range collapse toward a constant value.

For an interactive exploration open `identity_failure.ipynb`.  The notebook trains
standard MLPs, residual MLPs, and a stack of Neural Accumulator (NAC) layers across a
suite of activation functions (ReLU, ReLU6, Softplus, Tanh, Sigmoid, ELU, SiLU,
Identity, and PReLU) and visualises all models on a single log-scaled error plot.  This
plot makes it easy to see the subtle differences between activations that keep the error
extremely low while highlighting the dramatic failures of others.

## Configuration

The experiment parameters are collected in `ExperimentConfig` within
`identity_failure.py`.  You can modify the training range, depth, width, or number of
training points to explore how these choices affect the extrapolation performance.

The NAC and Neural Arithmetic Logic Unit (NALU) implementations used in both the script
and the notebook reside in `nac.py`.  They expose a `reset_parameters()` helper that
keeps the constrained weight matrix close to the additive regime recommended by the
original authors.
