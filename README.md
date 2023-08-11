# _Goras_: a _Keras_-like package for Go
_Goras_ is a package that aims to perform a similar function to _Keras_ in python. It is a somewhat high level package that contains code for making neural models easier than it would be when using the graph computation package directly. Just as _Keras_ uses tensorflow to actually perform maths, _Goras_ uses the excellent [_Gorgonia_](https://gorgonia.org) package for it's graph computation.

I am trying to design _Goras_ to have a similar workflow to the _Keras_ functional API. The functional API is capable of building of very complex models with relatively simple code. I have also tried to utilise types and functions from _Gorgonia_ wherever possible. This means that if somthing is not implemented yet, it is easy to add it yourself. That being said, if you do create any new layers, activations, or anything else, please feel free to pull request :).

## Stability
This is very much an unstable package. I am still trying to figure out how everything fits together best, so I will likely change function declarations and types quite a bit. That being said, the package is currently in a usable state, but just remember to tag onto a specific version.

## Features
- Similar workflow to Keras functional API
- Easy to build complex models with custom components
- Supports multiple model inputs and soon will support multiple model outputs
- Provides easy model saving and loading
- Supports many types of layers including Dense, Convolution2D, and MaxPooling2D
  - I plan to add support for LSTM and MHA layers in the future
## Examples
See the examples directory for some full examples. More coming soon! Alternatively, below is an overview of how the package works:

### Imports
```go
import (
	G "gorgonia.org/gorgonia"
	K "github.com/JoshPattman/goras"
	T "gorgonia.org/tensor"
)
```

### Building a model
```go
// We are going to use 4 as batch size as there are 4 rows in our dataset
batchSize := 4
// Define the topology
inputNodes, hiddenNodes, outputNodes := 2, 5, 1

// Create the empty model and a Namer to provide names to the layers
model := K.NewModel()
n := K.NewNamer("model")

// Create the input layer
inputs := K.Input(model, n.Next(), batchSize, inputNodes).Node
// Create the first Dense layer and its activation.
// Note that dense layers do not have an activation themselves, so you have to add one manually after
outputs := K.Dense(model, n.Next(), hiddenNodes).MustAttach(inputs)
outputs = K.Activation(model, n.Next(), "sigmoid").MustAttach(outputs)
// Create the second Dense layer
outputs = K.Dense(model, n.Next(), outputNodes).MustAttach(outputs)
outputs = K.Activation(model, n.Next(), "sigmoid").MustAttach(outputs)

// Build the rest of the model so we can train it and run it
// We are providing it with a mean squared error loss
model.MustBuild(K.WithInputs(inputs), K.WithOutputs(outputs), K.WithLosses(K.MSE))
```

### Fitting the model
```go
// Create an ADAM solver - this is the thing that actually updates the weights
solver := G.NewAdamSolver(G.WithLearnRate(0.01))
// Fit the model for 1k epochs
model.Fit(K.V(x), K.V(y), solver, K.WithEpochs(1000), K.WithLoggingEvery(100))
```

### Testing the model
```go
yp, _ := model.PredictBatch(K.V(x))
fmt.Printf("\nPredictions (%s):\n", testName)
for i := 0; i < x.Shape()[0]; i++ {
  sx, _ := x.Slice(T.S(i))
  sy, _ := y.Slice(T.S(i))
  syp, _ := yp.Slice(T.S(i))
  fmt.Printf("X=%v Y=%v YP=%.3f\n", sx, sy, syp)
}
```

## Todo
- Figure out how to allow models to have multiple outputs
  - I think just creating loss nodes for each then summing those nodes for a total loss is the way to go
- Add these layers
  - `Recurrent`
  - `LSTM`
  - `Convolution` - Simple implentation already
  - `Pooling` - Simple implentation already
  - `Deconvolution` - I think I will have to implement this in Gorgonia and pull request it first
  - `Upsampling` - I think I will have to implement this in Gorgonia and pull request it first
  - `Embedding`
- Add `L1` and `L2` regularlization
- Check if GPU support is working for cuda. I think it should work, but I havn't got round to testing yet.
- Currently, batching for training and prediction discards the remainder of the last batch (eg batch size 8, 17 elements, will only predict 16 things and the last thing will be disacrded).
  - I will fix this once I hear back on an issue https://github.com/gorgonia/gorgonia/issues/204
- Test and fix softmax and/or CCE
- Add callbacks for `Fit`
- Add Summary method to make visualising your network easier
  - Add functions to the layer interface that return previous layer (? not sure how to do this: maby store the previous nodes and then have a function in model to lookup nodes. Or maby name every node made by the layer constructors (eg previous_node = model_1(dense).matmul where the matmul node gets called model_1(dense).matmul))
  - Add functions to layer that return expected input and output shape