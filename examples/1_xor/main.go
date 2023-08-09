package main

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

	// Create the model
	model := MakeModel()

	// Test the model with no training
	TestModel(model, x, y, "no training")

	// Create an ADAM solver - this is the thing that actually updates the weights
	solver := G.NewAdamSolver(G.WithLearnRate(0.01))

	// Train the model for 1000 epochs
	for epoch := 0; epoch <= 1000; epoch++ {
		loss := model.FitBatch(x, y, solver)
		if epoch%100 == 0 {
			fmt.Printf("Epoch: %-4v Loss %.4f\n", fmt.Sprint(epoch), loss)
		}
	}

	// Test the model with after training
	TestModel(model, x, y, "after training")

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
	outputs := K.Dense(model, n.Next(), hiddenNodes).Attach(inputs)
	outputs = K.Activation(model, n.Next(), "sigmoid").Attach(outputs)
	// Create the second Dense layer
	outputs = K.Dense(model, n.Next(), outputNodes).Attach(outputs)
	outputs = K.Activation(model, n.Next(), "sigmoid").Attach(outputs)

	// Build the rest of the model so we can train it and run it
	// We are providing it with a mean squared error loss
	model.Build(inputs, outputs, K.MSE)

	return model
}

// Function to run a model and print some values to the terminal
func TestModel(model *K.Model, x, y *T.Dense, testName string) {
	// Test the model without any training
	ypNoTraining := model.PredictBatch(x)
	fmt.Printf("\nPredictions (%s):\n", testName)
	for i := 0; i < x.Shape()[0]; i++ {
		sx, _ := x.Slice(T.S(i))
		sy, _ := y.Slice(T.S(i))
		syp, _ := ypNoTraining.Slice(T.S(i))
		fmt.Printf("X=%v Y=%v YP=%.3f\n", sx, sy, syp)
	}
	fmt.Println()
}
