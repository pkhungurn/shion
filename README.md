# shion

`shion` (シオン) is my personal machine learning experimentation, designed to work with PyTorch.

## What does it do?

On top of machine learning frameworks such as [TensorFlow](https://www.tensorflow.org), [PyTorch](https://pytorch.org), 
and [JAX](https://github.com/google/jax), you need something like [fast.ai](https://www.fast.ai/) or 
[PyTorch Lightning](https://www.pytorchlightning.ai/) to make training and inference easy. Something like this should 
give you abstractions for:

* the evaluation of loss functions, 
* the invocation of optimizers, 
* the saving of snapshots, 
* the resumption from previous training run, 
* the logging of training and validation losses, and
* the visualization of your model’s outputs once in a while.

`shion` provides them in such a way that is hell-bent on separation of concerns and customization. I have stuck with 
the library in the past since 2019 without making many changes to the core abstractions. The caveat, though, is that 
it is quite hard to use even for myself.

## Installation

You can just copy the `src\shion` directory to your codebase, or you can use the following tools.

### Pip

```
pip install git+https://github.com/pkhungurn/shion.git
```

### Poetry

```
poetry add git+https://github.com/pkhungurn/shion.git
```

## Update History

* (2022/01/05) v0.1.0: First release.