# _Goras_: a _Keras_-like package for Go
_Goras_ is a package that aims to perform a similar function to _Keras_ in python. It is a somewhat high level package that contains code for making neural models easier than it would be when using the graph computation package directly. Just as _Keras_ uses tensorflow to actually perform maths, _Goras_ uses the excellent [_Gorgonia_](https://gorgonia.org) package for it's graph computation.

I am trying to design _Goras_ to have a similar workflow to the _Keras_ functional API. The functional API is capable of building of very complex models with relatively simple code. I have also tried to utilise types and functions from _Gorgonia_ wherever possible. This means that if somthing is not implemented yet, it is easy to add it yourself. That being said, if you do create any new layers, activations, or anything else, please feel free to pull request :).

## Stability
This is very much an unstable package. I am still trying to figure out how everything fits together best, so I will likely change function declarations and types quite a bit. That being said, the package is currently in a usable state, but just remember to tag onto a specific version.

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
outputs := K.Dense(model, n.Next(), hiddenNodes).Attach(inputs)
outputs = K.Activation(model, n.Next(), "sigmoid").Attach(outputs)
// Create the second Dense layer
outputs = K.Dense(model, n.Next(), outputNodes).Attach(outputs)
outputs = K.Activation(model, n.Next(), "sigmoid").Attach(outputs)

// Build the rest of the model so we can train it and run it
// We are providing it with a mean squared error loss
model.Build(inputs, outputs, K.MSE)
```

### Fitting the model
```go
// Create an ADAM solver - this is the thing that actually updates the weights
solver := G.NewAdamSolver(G.WithLearnRate(0.01))

// Train the model for 1000 epochs
for epoch := 0; epoch <= 1000; epoch++ {
    loss := model.FitBatch(x, y, solver)
    if epoch%100 == 0 {
        fmt.Printf("Epoch: %-4v Loss %.4f\n", fmt.Sprint(epoch), loss)
    }
}
```

### Testing the model
```go
yp := model.PredictBatch(x)
for i := 0; i < x.Shape()[0]; i++ {
    sx, _ := x.Slice(T.S(i))
    sy, _ := y.Slice(T.S(i))
    syp, _ := yp.Slice(T.S(i))
    fmt.Printf("X=%v Y=%v YP=%.3f\n", sx, sy, syp)
}
```

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
- Add these activations
  - `leaky_relu`
  - `tanh`
  - `binary`
- Add a `Fit` method to `Model` that splits the input into batch sized chunks, and shows a little loading bar
  - A `FitBatch` method already exists, but this requires the user to write the epochs loop and split the data into batches
- Add a `Predict` method to `Model` that splits the input into batches
- Figure out how to set certain layers in a model as trainable or not trainable
- Check if GPU support is working for cuda. I think it should work, but I havn't got round to testing yet.
- Better error handling
- Better error checking