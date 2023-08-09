# Goras: a Keras-like package for Go
Goras is a package that aims to perform a similar function to Keras in python. It is a somewhat high level package that contains code for making neural models easier than it would be when using the graph computation package directly. Just as Keras uses tensorflow to actually perform maths, Goras uses the excellent (Gorgonia)[https://gorgonia.org] package for it's graph computation.

I am trying to design Goras to have a similar workflow to the Keras functional API. The functional API is capable of building of very complex models with relatively simple code. I have also tried to utilise types and functions from Gorgonia wherever possible. This means that if somthing is not implemented yet, it is easy to add it yourself. That being said, if you do create any new layers, activations, or anything else, please feel free to pull request :).

## Examples
See the examples directory for some examples. More coming soon!

## Stability
This is very much an unstable package. I am still trying to figure out how everything fits together best, so I will likely change function declarations and types quite a bit. That being said, the package is currently in a usable state, but just remember to tag onto a specific version.

## Todo
- Figure out how to allow models to have multiple inputs and outputs
- Add these loss functions
  - `BCE`
  - `CCE`
- Add these layers
  - `Recurrent`
  - `LSTM`
  - `Convolution`
  - `Pooling`
  - `Deconvolution`
  - `Upsampling`
  - `Reshape`
- Add a `Fit` method to `Model` that splits the input into batch sized chunks, and shows a little loading bar
  - A `FitBatch` method already exists, but this requires the user to write the epochs loop and split the data into batches
- Add a `Predict` method to `Model` that splits the input into batches
- Figure out how to set certain layers in a model as trainable or not trainable
- Check if GPU support is working for cuda. I think it should work, but I havn't got round to testing yet.