// This is a basic example of using goras to create a simple neural network to solve the XOR problem.
// We will start by defining the dataset, then creating the model, summarising it, training it, and finally testing it.
// We will also then save the model to a file, and load it again, then test the loaded model.
package main

import (
	"fmt"
	"os"

	K "github.com/JoshPattman/goras" // I am using K here as I am used to using K for Keras, and G was already taken by Gorgonia
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func main() {
	// Create the dataset
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

	// Lets have a look at the model. The summary function generates a string to show you the layers, shapes, and connections.
	// It is extremely useful for debugging shapes and trying to figure out which parts of your neural network are causing it to be slowest due to them having too many weights.
	fmt.Printf("\nModel Summary:\n%s\n", model.Summary())
	/*OUTPUT
	Model Summary:
	Layer 0     model_1::input                Shape: (4, 2)               From: [] Num Params 0
	Layer 1     model_2::dense                Shape: (4, 5)               From: [model_1.input       ] Num Params 15
	Layer 2     model_3::activation(sigmoid)  Shape: (4, 5)               From: [model_2.matmul      ] Num Params 0
	Layer 3     model_4::dense                Shape: (4, 1)               From: [model_3.activation  ] Num Params 6
	Layer 4     model_5::activation(sigmoid)  Shape: (4, 1)               From: [model_4.matmul      ] Num Params 0
	Total number of parameters: 21
	*/

	// Test the model with no training
	TestModel(model, x, y, "no training")

	/*OUTPUT
	Predictions (no training):
	X=[0  0] Y=0 YP=0.727
	X=[0  1] Y=1 YP=0.657
	X=[1  0] Y=1 YP=0.755
	X=[1  1] Y=0 YP=0.686
	*/

	// Create an ADAM solver - this is the thing that actually updates the weights.
	// The solvers we use actually come with the Gorgonia package, and are not part of goras.
	solver := G.NewAdamSolver(G.WithLearnRate(0.01))

	// Train the model for 1000 epochs. We are also only going to log every 100 epochs, otherwise our terminal will be spammed with output.
	model.Fit(K.V(x), K.V(y), solver, K.WithEpochs(1000), K.WithLoggingEvery(100))

	/*OUTPUT
	Epoch 1/1000 - Loss: 0.294247 |Done|
	Epoch 100/1000 - Loss: 0.248415 |Done|
	Epoch 200/1000 - Loss: 0.210246 |Done|
	Epoch 300/1000 - Loss: 0.121424 |Done|
	Epoch 400/1000 - Loss: 0.041336 |Done|
	Epoch 500/1000 - Loss: 0.017041 |Done|
	Epoch 600/1000 - Loss: 0.009531 |Done|
	Epoch 700/1000 - Loss: 0.006241 |Done|
	Epoch 800/1000 - Loss: 0.004468 |Done|
	Epoch 900/1000 - Loss: 0.003386 |Done|
	Epoch 1000/1000 - Loss: 0.002669 |Done|
	*/

	// Test the model with after training. We can see that it is quite accurately predicting the XOR function.
	TestModel(model, x, y, "after training")

	/*OUTPUT
	Predictions (after training):
	X=[0  0] Y=0 YP=0.015
	X=[0  1] Y=1 YP=0.932
	X=[1  0] Y=1 YP=0.967
	X=[1  1] Y=0 YP=0.069
	*/

	// Save the model parameters to a file. Currently, Goras can only save the weights, not the structure of the model itself.
	// This means that if you want to use the model somewhere else, you will have to copy the MakeModel() function to that file.
	// We are also saving as a .gob file, as that is the format that Gorgonia uses for tensors.
	file, err := os.Create("./model.gob")
	if err != nil {
		panic(err)
	}
	model.WriteParams(file)
	file.Close()

	// Load a new model using the file we just created.
	loadedModel := MakeModel()
	file, err = os.Open("./model.gob")
	if err != nil {
		panic(err)
	}
	loadedModel.ReadParams(file)
	file.Close()

	// Test our loaded model. We can see that it is exactly the same as the model we just saved.
	TestModel(loadedModel, x, y, "after loading")

	/*OUTPUT
	Predictions (after loading):
	X=[0  0] Y=0 YP=0.015
	X=[0  1] Y=1 YP=0.932
	X=[1  0] Y=1 YP=0.967
	X=[1  1] Y=0 YP=0.069
	*/

	// The Predict() function of a Goras model allows you to put any number of samples in, and it will return the predictions for all of them.
	// It performs all the batching for you, so you don't have to worry about it.
	// Just remember that the input shape must be the same as the input shape of the model (except the batch dimension can be different).
	// Here we are testing with only half the dataset, which is 2 elements even though the batch size is 4.
	xSliced, _ := x.Slice(T.S(0, 2))
	ySliced, _ := y.Slice(T.S(0, 2))
	fmt.Println(xSliced.Shape())
	TestModel(loadedModel, xSliced, ySliced, "just two elements")

	/*OUTPUT
	(2, 2)

	Predictions (just two elements):
	X=[0  0] Y=0 YP=0.015
	X=[0  1] Y=1 YP=0.932
	*/

	// Lets also try with more elements than the batch size. This will be 6 elements, even though the batch size is 4.
	xConcated, _ := T.Concat(0, x, xSliced)
	yConcated, _ := T.Concat(0, y, ySliced)
	fmt.Println(xConcated.Shape())
	TestModel(loadedModel, xConcated, yConcated, "six elements")

	/*OUTPUT
	(6, 2)

	Predictions (six elements):
	X=[0  0] Y=0 YP=0.015
	X=[0  1] Y=1 YP=0.932
	X=[1  0] Y=1 YP=0.967
	X=[1  1] Y=0 YP=0.069
	X=[0  0] Y=0 YP=0.015
	X=[0  1] Y=1 YP=0.932
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
	// We are going to use 4 as batch size as there are 4 rows in our dataset.
	// There is currently an issue with Goras that means that when batching the data for training,
	// if the final batch is smaller than the others (because the dataset size is not divisible by the batch size), the last batch will be ignored.
	// So for example, if we tried to use a batch size of 3, our model would only train on the first 3 elements in the dataset.
	// However, if we set the batch size to 2, all 4 elements could be trained on again.
	// This is not an issue for prediction, as the input data is padded with 0s to make it the correct size.
	batchSize := 4
	// Define the topology of the model. We are using 2 input nodes, 5 hidden nodes, and 1 output node.
	inputNodes, hiddenNodes, outputNodes := 2, 5, 1

	// Create the empty model with the Float64 tensor type. In Goras, all layers in a model must share the same tensor type (for now at least).
	model := K.NewModel(T.Float64)

	// Create a function to give each layer a unique name. Names should be unique per layer and are used for debugging (summary) and save/load.
	// Goras provides the K.NewNamer() function to help with this. It returns a function that generates a new name each time it is called.
	// If the base name is "model", the generated names will be "model_1", "model_2", etc.
	n := K.NewNamer("model")

	// Create the input layer. This is how we tell our model the input shape.
	// Notice how we have to define the batch size here. For now, models have a fixed batch size.
	// Batch size should always be the first element of the input shape e.g. (batch_size, other_dims...)
	inputs := K.Input(model, n(), batchSize, inputNodes).Node()
	// Create the first Dense (fully connected) layer and its activation.
	// Note that dense layers do not have an activation themselves, so you have to add one manually after
	outputs := K.Dense(model, n(), hiddenNodes).MustAttach(inputs)
	// We will use sigmoid activations for the hidden layers
	outputs = K.Activation(model, n(), "sigmoid").MustAttach(outputs)
	// Create the second Dense layer
	outputs = K.Dense(model, n(), outputNodes).MustAttach(outputs)
	// This is the other way you can add activations. They are both exactly equivalent.
	outputs = K.Sigmoid(model, n()).MustAttach(outputs)

	// In Goras, building the model is the process of creating the loss, output, and other nodes that were not created in the previous steps.
	// Building the model also creates the *G.TapeMachine that is used to run the model.
	// We are going to use the mean squared error loss function.
	// Goras models also support multiple inputs and outputs.
	// We are only using one of each here, but if you wanted to use multiple outputs, you should also remeber to specify the loss for each output.
	model.MustBuild(K.WithInputs(inputs), K.WithOutputs(outputs), K.WithLosses(K.MSE))

	return model
}

// Function to run a model and print some values to the terminal
func TestModel(model *K.Model, x, y T.Tensor, testName string) {
	// Predict the output for the given input.
	// As our model expects a slice of inputs (to support multiple input models), we need to wrap our input in a slice.
	// K.V(x) simply is shorthand for []T.Tensor{x}
	yps, err := model.Predict(K.V(x))
	if err != nil {
		panic(err)
	}
	// Again, the predict funtion returns a slice of outputs, but as we only have a single output node, need to get the first one.
	yp := yps[0]

	fmt.Printf("\nPredictions (%s):\n", testName)
	for i := 0; i < yp.Shape()[0]; i++ {
		sx, _ := x.Slice(T.S(i))
		sy, _ := y.Slice(T.S(i))
		syp, _ := yp.Slice(T.S(i))
		fmt.Printf("X=%v Y=%v YP=%.3f\n", sx, sy, syp)
	}
	fmt.Println()
}
