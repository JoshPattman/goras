[![Go Reference](https://pkg.go.dev/badge/github.com/JoshPattman/goras.svg)](https://pkg.go.dev/github.com/JoshPattman/goras)
[![Go Coverage](https://github.com/JoshPattman/goras/wiki/coverage.svg)](https://raw.githack.com/wiki/JoshPattman/goras/coverage.html)
![CI Status](https://github.com/JoshPattman/goras/actions/workflows/go.yml/badge.svg)
[![Stability](http://badges.github.io/stability-badges/dist/unstable.svg)](http://github.com/badges/stability-badges)
[![Go Report](https://goreportcard.com/badge/github.com/JoshPattman/goras)](https://goreportcard.com/badge/github.com/JoshPattman/goras)

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
- Supports fitting models with data generators
- Supports multiple types of layers, with more on the way
  - `Dense`
  - `Conv2D`
  - `MaxPooling2D`
  - `Dropout`
  - `Reshape`
  - `OneHot`
- Supports many loss functions with a very flexible method of adding more
  - `Mean Squared Error`
  - `Binary Cross-Entropy`
  - `Categorical Cross-Entropy`
  - `L2 Normalisation`
  - `Weighted Additive Loss` - For combining multiple losses for multiple outputs
## Examples
The `examples/` directory contains multiple examples, with detailed comments throughout explaining each step. It is recommended that you read through the examples in order, as most concepts are only talked about once. Alternatively, below are some short code snippets using **Goras**. Note that in these examples, many methods are named `MustXXX(...)`, which means that **Goras** will run the function `XXX()` which returns an some data and an error, but will only return the data. It will panic if an error occurs.
### Build a model
Building a model is simple in Goras, however the api can easily allow you to build complex models, far beyond what sequential model building can produce.

```go
batchSize := 4
inputNodes, hiddenNodes, outputNodes := 2, 5, 1

model := K.NewModel()

n := K.NewNamer("model")

inputs := K.Input(model, n(), T.Float64, batchSize, inputNodes).Node()
outputs := K.Dense(model, n(), hiddenNodes).MustAttach(inputs)
outputs = K.Activation(model, n(), "sigmoid").MustAttach(outputs)
outputs = K.Dense(model, n(), outputNodes).MustAttach(outputs)
outputs = K.Sigmoid(model, n()).MustAttach(outputs)

model.MustBuild(K.WithInput("x", inputs), K.WithOutput("yp", outputs), K.WithLoss(K.MSELoss("yt", outputs)))

return model
```

### Fit a model to data
Fitting a model in **Goras** requires just one line of code. The `Fit` method is extensible, using constructor options. **Goras** also supports data generators, which allow data to be loaded one batch at a time, instead of all before `Fit is called`
```go
model.MustFit(K.NamedTs{"x": x}, K.NamedTs{"yt": y}, solver, K.WithEpochs(1000), K.WithLoggingEvery(100))
```

### Predicting with a model
Predicting using a model is just as simple as fitting.
```go
outs := model.MustPredict(K.NamedTs{"x": x})
yp := outs["yp"]
```

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
