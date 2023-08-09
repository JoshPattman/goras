package main

// In this example, we see how to create a goras model, and use it to approximate the XOR function.
// We also save and load the model from disk.

import (
	"fmt"
	"os"

	G "gorgonia.org/gorgonia"

	K "github.com/JoshPattman/goras"
	T "gorgonia.org/tensor"
)

func main() {
	// Create the data
	x, y := LoadXY()

	/*OUTPUT
	X:
	⎡0  0⎤
	⎢0  1⎥
	⎢1  0⎥
	⎣1  1⎦

	Y:
	C[0  1  1  0]
	*/

	// Create the model
	model := MakeModel()

	// Test the model with no training
	TestModel(model, x, y, "no training")

	/*OUTPUT
	Predictions (no training):
	X=[0  0] Y=0 YP=0.799
	X=[0  1] Y=1 YP=0.799
	X=[1  0] Y=1 YP=0.770
	X=[1  1] Y=0 YP=0.769
	*/

	// Create an ADAM solver - this is the thing that actually updates the weights
	solver := G.NewAdamSolver(G.WithLearnRate(0.01))

	// Train the model for 1000 epochs
	model.Fit(x, y, solver, K.WithEpochs(1000), K.WithLoggingEvery(100))

	/*OUTPUT
	Epoch: 0    Loss 0.3305
	Epoch: 100  Loss 0.2438
	Epoch: 200  Loss 0.1999
	Epoch: 300  Loss 0.1023
	Epoch: 400  Loss 0.0305
	Epoch: 500  Loss 0.0124
	Epoch: 600  Loss 0.0068
	Epoch: 700  Loss 0.0043
	Epoch: 800  Loss 0.0031
	Epoch: 900  Loss 0.0023
	Epoch: 1000 Loss 0.0018
	*/

	// Test the model with after training
	TestModel(model, x, y, "after training")

	/*OUTPUT
	Predictions (after training):
	X=[0  0] Y=0 YP=0.042
	X=[0  1] Y=1 YP=0.961
	X=[1  0] Y=1 YP=0.953
	X=[1  1] Y=0 YP=0.040
	*/

	// Save the model to a file
	file, err := os.Create("./model.gob")
	if err != nil {
		panic(err)
	}
	model.WriteParams(file)
	file.Close()

	// Load a new model using that file
	// Note that currently, in goras, you must create the model with code first, and the only thing that is actually loaded are the weights.
	// This means that if you wanted to use this model in another file, you would have to copy the MakeModel() function to that file.
	loadedModel := MakeModel()
	file, err = os.Open("./model.gob")
	if err != nil {
		panic(err)
	}
	loadedModel.ReadParams(file)
	file.Close()

	// Test our loaded model
	TestModel(loadedModel, x, y, "after loading")

	/*OUTPUT
	Predictions (after loading):
	X=[0  0] Y=0 YP=0.042
	X=[0  1] Y=1 YP=0.961
	X=[1  0] Y=1 YP=0.953
	X=[1  1] Y=0 YP=0.040
	*/
}

// Function to create the X and Y data as tensors
func LoadXY() (*T.Dense, *T.Dense) {
	x := T.New(
		T.WithShape(4, 2),
		T.WithBacking([]float64{0, 0, 0, 1, 1, 0, 1, 1}),
	)
	y := T.New(
		T.WithShape(4, 1),
		T.WithBacking([]float64{0, 1, 1, 0}),
	)
	fmt.Printf("X:\n%v\n", x)
	fmt.Printf("Y:\n%v\n", y)
	return x, y
}

// Function to create a new model with randomized weights
func MakeModel() *K.Model {
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
	model.MustBuild(inputs, outputs, K.MSE)

	return model
}

// Function to run a model and print some values to the terminal
func TestModel(model *K.Model, x, y *T.Dense, testName string) {
	// Test the model without any training
	ypNoTraining, _ := model.PredictBatch(x)
	fmt.Printf("\nPredictions (%s):\n", testName)
	for i := 0; i < x.Shape()[0]; i++ {
		sx, _ := x.Slice(T.S(i))
		sy, _ := y.Slice(T.S(i))
		syp, _ := ypNoTraining.Slice(T.S(i))
		fmt.Printf("X=%v Y=%v YP=%.3f\n", sx, sy, syp)
	}
	fmt.Println()
}
