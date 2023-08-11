package main

// In this example, we create a model to classify MNIST digits.
// It uses convolution2d and maxpool2d layers, and it is the same underlying model as https://gorgonia.org/tutorials/mnist/

import (
	"fmt"
	"time"

	K "github.com/JoshPattman/goras"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// We will define a dtype. Float32 in theory should be faster but I haven't seen much difference. It should also occupy less memory though.
var dType = T.Float32

func main() {
	// Lets start by loading MNIST into tensors
	x, y, err := Load("train", "./mnist", dType)
	if err != nil {
		panic(err)
	}
	fmt.Println("x:", x.Shape())
	fmt.Println("y:", y.Shape())

	/*OUTPUT
	x: (60000, 784)
	y: (60000, 10)
	*/

	// We need to reshape the x data to be (batch_size, channels, img_x, img_y)
	x.Reshape(60000, 1, 28, 28)
	fmt.Println("x:", x.Shape())

	/* OUTPUT:
	x: (60000, 1, 28, 28)
	*/

	// Create the model
	model := MakeModel()

	// Lets use an ADAM solver again
	solver := G.NewAdamSolver(G.WithLearnRate(0.001))

	// Fit the model. You only need about 3 epochs as mnist is quite simple.
	// We also specify not to clear the line, which means we can see the progress.
	fitStart := time.Now()
	err = model.Fit(K.V(x), K.V(y), solver, K.WithClearLine(false), K.WithEpochs(3))
	if err != nil {
		panic(err)
	}
	fmt.Printf("Done Training. It took %v\n", time.Since(fitStart))

	/* OUTPUT:
	Epoch 1/3 - Loss: 0.001303
	Epoch 2/3 - Loss: 0.000731
	Epoch 3/3 - Loss: 0.000648
	*/

	// Now we can predict on the first 16 images
	// For now, we need to predict a multiple of the batch size (which is 16)
	fmt.Println("\nPredictions:")

	xB, _ := x.Slice(T.S(0, 16))
	yB, _ := y.Slice(T.S(0, 16))
	yBP, err := model.PredictBatch(K.V(xB))
	if err != nil {
		panic(err)
	}
	// Now for each prediction, we can get the argmax to see what the model predicted
	// We can then print that out along with the actual label to see if it was right
	for i := 0; i < 16; i++ {
		yp, _ := yBP.Slice(T.S(i))
		yb, _ := yB.Slice(T.S(i))
		predicted, _ := T.Argmax(yp, 0)
		actual, _ := T.Argmax(yb, 0)
		fmt.Println("\tPrediction:", predicted)
		fmt.Println("\tActual:", actual)
		fmt.Println()
	}

	// Notice how well the model did! It got all but one correct.
	// This may not be the case for you, but it should be pretty close.
	/*OUTPUT:
	Predictions:
		Prediction: 3
		Actual: 5

		Prediction: 0
		Actual: 0

		Prediction: 4
		Actual: 4

		Prediction: 1
		Actual: 1

		Prediction: 9
		Actual: 9

		Prediction: 2
		Actual: 2

		Prediction: 1
		Actual: 1

		Prediction: 3
		Actual: 3

		Prediction: 1
		Actual: 1

		Prediction: 4
		Actual: 4

		Prediction: 3
		Actual: 3

		Prediction: 5
		Actual: 5

		Prediction: 3
		Actual: 3

		Prediction: 6
		Actual: 6

		Prediction: 1
		Actual: 1

		Prediction: 7
		Actual: 7
	*/
}

// Function to create the same model as descibed in https://gorgonia.org/tutorials/mnist/
func MakeModel() *K.Model {
	// Lets define a batch size of 16 for now
	batchSize := 16

	// Create the model and namer
	model := K.NewModel(dType)
	n := K.NewNamer("model")

	// Input shape is (batch_size, channels(this is one for b&w), img_x, img_y)
	inputs := K.Input(model, n.Next(), batchSize, 1, 28, 28).Node()

	// Convolution and pooling blocks
	outputs := K.SimpleConv2D(model, n.Next(), 3, 32).MustAttach(inputs)
	outputs = K.Activation(model, n.Next(), "relu").MustAttach(outputs)
	outputs = K.SimpleMaxPooling2D(model, n.Next(), 2).MustAttach(outputs)
	outputs = K.Dropout(model, n.Next(), 0.2).MustAttach(outputs)

	outputs = K.SimpleConv2D(model, n.Next(), 3, 64).MustAttach(outputs)
	outputs = K.Activation(model, n.Next(), "relu").MustAttach(outputs)
	outputs = K.SimpleMaxPooling2D(model, n.Next(), 2).MustAttach(outputs)
	outputs = K.Dropout(model, n.Next(), 0.2).MustAttach(outputs)

	outputs = K.SimpleConv2D(model, n.Next(), 3, 128).MustAttach(outputs)
	outputs = K.Activation(model, n.Next(), "relu").MustAttach(outputs)
	outputs = K.SimpleMaxPooling2D(model, n.Next(), 2).MustAttach(outputs)

	// Reshape and dropout
	b, c, h, w := outputs.Shape()[0], outputs.Shape()[1], outputs.Shape()[2], outputs.Shape()[3]
	outputs = K.Reshape(model, n.Next(), T.Shape{b, c * h * w}).MustAttach(outputs)
	outputs = K.Dropout(model, n.Next(), 0.2).MustAttach(outputs)

	// Dense layers
	outputs = K.Dense(model, n.Next(), 625).MustAttach(outputs)
	outputs = K.Activation(model, n.Next(), "relu").MustAttach(outputs)
	outputs = K.Dropout(model, n.Next(), 0.55).MustAttach(outputs)
	outputs = K.Dense(model, n.Next(), 10).MustAttach(outputs)
	// For now we are using sigmoid as the activation. Softmax would be better but I think at the moment it is broken.
	outputs = K.Activation(model, n.Next(), "sigmoid").MustAttach(outputs)

	// Build the model
	// Again, we are using MSE as the loss function. CCE would be better if we were using softmax.
	model.MustBuild(K.WithInputs(inputs), K.WithOutputs(outputs), K.WithLosses(K.MSE))

	return model
}
