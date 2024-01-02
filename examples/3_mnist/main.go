// In this example, we create a model to classify MNIST digits.
// It uses convolution2d and maxpool2d layers, and it is a similar underlying model as https://gorgonia.org/tutorials/mnist/
// the only difference being that this model uses fewer conv filters and a smaller dense layer, for speed purposes.
// We will also use dropout layers, and use a save model callback in the fit function to save the model after every epoch.
package main

import (
	"fmt"
	"time"

	K "github.com/JoshPattman/goras"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// We will define a dtype. Float32 occupies less memory than Float64.
var dType = T.Float32

func main() {
	// Lets start by loading MNIST into tensors. This code is explained in more detail in the Gorgonia tutorial.
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

	// See a summary of the model. You can see this model has many more parameters (numbers to change) than the XOR models.
	fmt.Printf("\nModel Summary:\n%s\n", model.Summary())

	// Lets use an ADAM solver again
	solver := G.NewAdamSolver(G.WithLearnRate(0.001))

	// Fit the model. You only need about 3 epochs as mnist is quite simple, and there are also a lot of digits.
	// When we run this, the fit method will by default show us a loading bar and some current information.
	// This can be turned off by adding K.WithVerbose(false) as one of the arguments.
	// For exemplar purposes, we also save the model parameters after each epoch using a callback.
	// You can also write your own callbacks very simply by making a function `func (epoch int) error`.
	fitStart := time.Now()
	err = model.Fit(K.NamedTs{"input": x}, K.NamedTs{"output_target": y}, solver, K.WithClearLine(false), K.WithEpochs(3), K.WithEpochCallbacks(K.SaveModelParametersCallback(model, "./model.gob")))
	if err != nil {
		panic(err)
	}
	fmt.Printf("Done Training. It took %v\n", time.Since(fitStart))

	// On my not very powerful laptop, this took ~5 minutes.
	// I have not tested it yet, but I hope to one day allow Goras to use GPUs, which would speed up this process greatly.
	// This should be very feasable, as Gorgonia already supports some GPU operations.
	/* OUTPUT:
	Epoch 1/3 - Loss: 0.022752 |Done|
	Epoch 2/3 - Loss: 0.011735 |Done|
	Epoch 3/3 - Loss: 0.010036 |Done|

	Done Training. It took 4m38.525445874s
	*/

	// Now we can predict some images
	fmt.Println("\nPredictions:")
	numToPredict := 10
	xB, _ := x.Slice(T.S(0, numToPredict))
	yB, _ := y.Slice(T.S(0, numToPredict))
	yBsP, err := model.Predict(K.NamedTs{"input": xB})
	if err != nil {
		panic(err)
	}
	yBP := yBsP["output"]

	// For each prediction, our model outputs 10 numbers which correspond to how likely the image is that digit.
	//We can get the argmax to see what digit the model predicted was most likely.
	// We can then print that out along with the actual label to see if it was right.
	for i := 0; i < numToPredict; i++ {
		yp, _ := yBP.Slice(T.S(i))
		yb, _ := yB.Slice(T.S(i))
		predicted, _ := T.Argmax(yp, 0)
		actual, _ := T.Argmax(yb, 0)
		fmt.Println("\tPrediction:", predicted)
		fmt.Println("\tActual:", actual)
		fmt.Println()
	}

	// Notice how well the model did! It got all correct!
	// This may not be the case for you, but it should be pretty close.
	/*OUTPUT:
		Predictions:
	        Prediction: 5
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
	*/
}

// Function to create a similar model as descibed in https://gorgonia.org/tutorials/mnist/
func MakeModel() *K.Model {
	batchSize := 16

	model := K.NewModel()
	n := K.NewNamer("model")

	// Input shape is (batch_size, channels(this is 1 for b&w), img_x, img_y)
	inputs := K.Input(model, n(), dType, batchSize, 1, 28, 28).Node()

	// Convolution and pooling blocks
	outputs := K.SimpleConv2D(model, n(), 3, 16).MustAttach(inputs)
	outputs = K.Activation(model, n(), "relu").MustAttach(outputs)
	outputs = K.SimpleMaxPooling2D(model, n(), 2).MustAttach(outputs)
	// A Dropout layer simply zeros out some randomly selected values. This can help with overfitting.
	outputs = K.Dropout(model, n(), 0.2).MustAttach(outputs)

	outputs = K.SimpleConv2D(model, n(), 3, 32).MustAttach(outputs)
	outputs = K.Activation(model, n(), "relu").MustAttach(outputs)
	outputs = K.SimpleMaxPooling2D(model, n(), 2).MustAttach(outputs)
	outputs = K.Dropout(model, n(), 0.2).MustAttach(outputs)

	outputs = K.SimpleConv2D(model, n(), 3, 32).MustAttach(outputs)
	outputs = K.Activation(model, n(), "relu").MustAttach(outputs)
	outputs = K.SimpleMaxPooling2D(model, n(), 2).MustAttach(outputs)

	// Reshape and dropout. We want to reshape from (batch_size, prev_filters, prev_height, prev_width) to (batch_size, prev_filters*prev_height*prev_width).
	// This flattens the data so a dense layer can understand it.
	b, c, h, w := outputs.Shape()[0], outputs.Shape()[1], outputs.Shape()[2], outputs.Shape()[3]
	outputs = K.Reshape(model, n(), T.Shape{b, c * h * w}).MustAttach(outputs)
	outputs = K.Dropout(model, n(), 0.2).MustAttach(outputs)

	// Dense layers
	outputs = K.Dense(model, n(), 64).MustAttach(outputs)
	outputs = K.Activation(model, n(), "relu").MustAttach(outputs)
	outputs = K.Dropout(model, n(), 0.55).MustAttach(outputs)
	outputs = K.Dense(model, n(), 10).MustAttach(outputs)
	// We are using softmax as the activation function for the last layer.
	outputs = K.Activation(model, n(), "softmax").MustAttach(outputs)

	// Build the model
	// This time we will use CCC (Categorical Cross Entropy) as the loss function. This must be used with softmax.
	model.MustBuild(K.WithInput("input", inputs), K.WithOutput("output", outputs), K.WithLoss(K.CCELoss("output_target", outputs)))

	return model
}
