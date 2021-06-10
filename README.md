# Masksembles for Uncertainty Estimation

### [Project Page](https://nikitadurasov.github.io/projects/masksembles/) | [Paper](https://arxiv.org/abs/2012.08334) | [Video Explanation](#)

[![Open Masksembles in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/masksembles/blob/main/notebooks/MNIST_Masksembles.ipynb)

---
Official implementation of Masksembles approach from the paper "Masksembles for Uncertainty Estimation" by 
Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua (CVPR 2021).
---
<p align="center">
  <img style="border-radius: 30px" src="https://raw.githubusercontent.com/nikitadurasov/masksembles/main/images/transition.gif" />
</p>

## Why Uncertainty?

TBD

## Installation

To install this package, use:

```bash
pip install git+http://github.com/nikitadurasov/masksembles
```

In addition, Masksembles requires installing at least one of the backends: torch or tensorflow2 / keras.
Please follow official installation instructions for [torch](https://pytorch.org/) or [tensorflow](https://www.tensorflow.org/install)
accordingly.


## Usage 

[comment]: <> (In masksembles module you could find implementations of "Masksembles{1|2|3}D" that)

[comment]: <> (support different shapes of input vectors &#40;1, 2 and 3-dimentional accordingly&#41;)

This package provides implementations for `Masksembles{1|2|3}D` layers in `masksembles.{torch|keras}` 
where `{1|2|3}` refers to dimensionality of input tensors (1-, 2- and 3-dimensional 
accordingly).

* `Masksembles1D`: works with 1-dim inputs,`[B, C]` shaped tensors
* `Masksembles2D`: works with 2-dim inputs,`[B, H, W, C]` (keras) or `[B, C, H, W]` (torch) shaped tensors
* `Masksembles3D` : TBD

In a Nutshell, Masksembles applies binary masks to inputs via multiplying them both channel-wise. For more efficient
implementation we've followed approach similar to [this](https://arxiv.org/abs/2002.06715) one. Therefore, after inference
`outputs[:B // N]` - stores results for the first submodel, `outputs[B // N : 2 * B // N]` - for the second and etc.  
### Torch 

```python 
import torch
from masksembles.torch import Masksembles1D

layer = Masksembles1D(10, 4, 2.)
layer(torch.ones([4, 10]))
```
```bash
tensor([[0., 1., 0., 0., 1., 0., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1., 0., 0., 1., 1.],
        [1., 0., 1., 1., 0., 0., 1., 0., 1., 1.],
        [1., 0., 0., 1., 1., 1., 0., 1., 1., 0.]], dtype=torch.float64)

```

### Tensorflow / Keras

```python 
import tensorflow as tf 
from masksembles.keras import Masksembles1D

layer = Masksembles1D(4, 2.)
layer(tf.ones([4, 10]))
```
```bash
<tf.Tensor: shape=(4, 10), dtype=float32, numpy=
array([[0., 1., 1., 0., 1., 1., 1., 0., 1., 0.],
       [0., 1., 0., 1., 1., 0., 1., 1., 0., 1.],
       [1., 1., 1., 1., 0., 0., 1., 0., 0., 1.],
       [1., 0., 0., 1., 0., 1., 1., 0., 1., 1.]], dtype=float32)>
```

### Model example
```python 
import tensorflow as tf 
from masksembles.keras import Masksembles1D, Masksembles2D

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="elu"),
        Masksembles2D(4, 2.0),
        layers.MaxPooling2D(pool_size=(2, 2)),
     
        layers.Conv2D(64, kernel_size=(3, 3), activation="elu"),
        Masksembles2D(4, 2.0),
        layers.MaxPooling2D(pool_size=(2, 2)),
     
        layers.Flatten(),
        Masksembles1D(4, 2.),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
```

## Citation
If you found this work useful for your projects, please don't forget to cite it.
```
@inproceedings{Durasov21,
  author = {N. Durasov and T. Bagautdinov and P. Baque and P. Fua},
  title = {{Masksembles for Uncertainty Estimation}},
  booktitle = CVPR,
  year = 2021
}
```