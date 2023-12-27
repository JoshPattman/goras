[![Go Reference](https://pkg.go.dev/badge/github.com/JoshPattman/goras.svg)](https://pkg.go.dev/github.com/JoshPattman/goras)
[![Go Coverage](https://github.com/JoshPattman/goras/wiki/coverage.svg)](https://raw.githack.com/wiki/JoshPattman/goras/coverage.html)
![CI Status](https://github.com/JoshPattman/goras/actions/workflows/go.yml/badge.svg)
[![unstable](http://badges.github.io/stability-badges/dist/unstable.svg)](http://github.com/badges/stability-badges)

# _Goras_: a _Keras_-like package for Go
_Goras_ is a package that aims to perform a similar function to _Keras_ in python. It is a somewhat high level package that contains code for making neural models easier than it would be when using the graph computation package directly. Just as _Keras_ uses tensorflow to actually perform maths, _Goras_ uses the excellent [_Gorgonia_](https://gorgonia.org) package for it's graph computation.

I am trying to design _Goras_ to have a similar workflow to the _Keras_ functional API. The functional API is capable of building of very complex models with relatively simple code. I have also tried to utilise types and functions from _Gorgonia_ wherever possible. This means that if something is not implemented yet, it is easy to add it yourself. That being said, if you do create any new layers, activations, or anything else, please feel free to pull request :).

## Stability
This is very much an unstable package. I am still trying to figure out how everything fits together best, so I will likely change function declarations and types quite a bit. That being said, the package is currently in a usable state, but just remember to tag onto a specific version.

## Features
- Workflow inspired by Keras functional API
- Easy to build complex models with custom components
- Supports multiple model inputs and outputs
- Provides easy model weights saving and loading
- Supports many types of layers including Dense, Convolution2D, and MaxPooling2D
  - I plan to add support for LSTM and MHA layers in the future
## Examples
See the examples directory for some full examples. More coming soon!
## Todo
- Add these layers
  - `Recurrent`
  - `LSTM`
  - `Deconvolution` - I think I will have to implement this in Gorgonia and pull request it first
  - `Upsampling`
  - `Embedding` - This could be done with a dense layer (like in golgi) but i think writing a custom embedding layer would be more efficient.
- Add `L1` and `L2` regularlization
- Check if GPU support is working for cuda. I think it should work, but I havn't got round to testing yet.
- Currently, batching for training discards the remainder of the last batch (eg batch size 8, 17 elements, will only fit 16 things and the last thing will be discarded).
  - I will fix this once I hear back on an issue https://github.com/gorgonia/gorgonia/issues/204
  - Batching for prediction zero pads but this is a bit wasteful
- Add more callbacks for `Fit`
- Add a shuffle parameter to fit
- Add `SCCE` Loss
  - I now know how to add this, and will do so when I hear if a onehot op will get added to gorgonia
  - https://github.com/gorgonia/gorgonia/issues/559
