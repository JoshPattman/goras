[![Go Reference](https://pkg.go.dev/badge/github.com/JoshPattman/goras.svg)](https://pkg.go.dev/github.com/JoshPattman/goras)
[![Go Coverage](https://github.com/JoshPattman/goras/wiki/coverage.svg)](https://raw.githack.com/wiki/JoshPattman/goras/coverage.html)
![CI Status](https://github.com/JoshPattman/goras/actions/workflows/go.yml/badge.svg)
[![unstable](http://badges.github.io/stability-badges/dist/unstable.svg)](http://github.com/badges/stability-badges)

# **Goras**: A high-level neural network package for Go
Goras is a Go package offering a high-level interface for simplifying neural network implementation, leveraging the excellent [**Gorgonia**](https://gorgonia.org) package for graph computation. **Goras** has a workflow akin to the [**Keras**](https://keras.io/) functional API, which is capable of building both simple and complex models with relatively simple code.

The package is built to be extensible, leveraging types and patterns from **Gorgonia** wherever possible. This means that if a feature does not exist yet, it should be trivial to implement it yourself.

## Stability
Though currently labeled as unstable, this package is still usable, with almost all features in working order. However, be aware that future changes may occur, mainly in response to the anticipated release of **Gorgonia** version `0.10`. While some adjustments might happen, the core functionality of **Goras** is expected to remain largely the same.

## Overview of Features
- Workflow inspired by **Keras** functional API
- Easy to build complex models with custom components
- Supports multiple model inputs and outputs
- Provides simple model weights saving and loading
- Supports multiple types of layers, with more on the way
  - _Dense_
  - _Conv2D_
  - _MaxPooling2D_
  - _Dropout_
  - _Reshape_
- Supports many loss functions with a very flexible method of adding more
  - _Mean Squared Error_
  - _Binary Cross-Entropy_
  - _Categorical Cross-Entropy_
  - _L2 Normalisation_
  - _Weighted Additive Loss_ - For combining multiple losses for multiple outputs
## Examples
The `examples/` directory contains multiple examples, with detailed comments throughout explaining each step. It is recommended that you read through the examples in order, as most concepts are only talked about once.
## Todo
- Add these layers (most of these will need to implement the op in gorgonia first)
  - `Recurrent`
  - `LSTM`
  - `Deconvolution`
  - `Upsampling`
  - `Embedding`
  - `MultiHeadAttention`
  - `BatchNorm`
  - `LayerNorm`
  - `Concat`
  - `OneHot`
- Increase test coverage
- Add `L1` regularlization
- Currently, batching for training discards the remainder of the last batch (eg batch size 8, 17 elements, will only fit 16 things and the last thing will be discarded).
  - I will fix this once I hear back on an issue https://github.com/gorgonia/gorgonia/issues/204
  - Batching for prediction zero pads but this is a bit wasteful
  - This is not really a big problem though, and if you really need inference performance for a certain batch size, you can just make another model and copy the weights over
- Add more callbacks for `Fit`
- Add a shuffle parameter to fit
- Tensorboard Integration
- Add `SCCE` Loss
  - I now know how to add this, and will do so when I hear if a onehot op will get added to gorgonia
  - https://github.com/gorgonia/gorgonia/issues/559
- Make a way to not only save model weights but also the model structure (not sure how to do this well yet though)
- Get GPU support working. I am waiting for gorgonia v0.10 for this as I think the new version changes a lot of CUDA stuff.
