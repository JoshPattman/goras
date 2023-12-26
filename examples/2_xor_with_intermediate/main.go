// In this example, we create two goras models which are linked, meaning changing the weights of one changes the weights of the other.
// We use this to view the intermediate layer values when performing an XOR operation.
// I think that this technique could be used to train more complex multi-section networks,
// such as autoencoders (sometimes want to just use encoder or just use decoder) and GANs (sometimes want to just generate).
// If you havn't already, please read through the first example (examples/1_xor/main.go) before reading this one, as I have not exmplained some things twice.
package main

import (
	"fmt"

	K "github.com/JoshPattman/goras"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func main() {
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

	// Lets create the main model - this is the same model used in example 1.
	// However, this time our MakeModel function is capable of creating a model with only the first layer, if you pass 'true' to it.
	modelFull := MakeModel(false)
	// Lets also assume that we want to know the values of the intermediate layer for some reason.
	// Currently, the way this works in Goras is to create a second network which shares the weights tensors with the main network by sharing the same underlying data.
	// This is quite fast, but you MUST make sure not to run two linked models at the same time.
	modelPartial := MakeModel(true)
	// Now we want to link the weights of the two models.
	// The weights are linked based on layer name, so it is important that each sub section of the network that can be used independantly is created with a different namer.
	// This works by formatting the layer name with the weight name (e.g. model_1.weights). This means if you want two layers in two models to share weights, they must have the same name.
	// The BindParamsFrom function will not modify the weights of the second network, but will instead point the weights of the first ('modelPartial') model to the data of the second ('modelFull') model.
	modelPartial.BindParamsFrom(modelFull)

	// Lets just see what the intermediate layer looks like without training
	TestModel(modelPartial, x, nil, "partial no training")

	/* OUTPUT
	Predictions (partial no training):
	X=[0  0] YP=[0.293  0.422  0.381  0.439  0.470]
	X=[0  1] YP=[0.294  0.331  0.065  0.374  0.672]
	X=[1  0] YP=[0.424  0.615  0.528  0.509  0.544]
	X=[1  1] YP=[0.425  0.520  0.112  0.442  0.734]
	*/

	// Ok lets now train the model. Note we only have to train the full model, and the partial model will be updated automatically.
	// Also this time we are using fitBatch instead of fit just to show you can do it either way.
	// Remeber though, FitBatch requires the data to be the exact same length as the batch size (in this case we chose a batch size of 4 anyway).
	// FitBatch also does not have some of the useful features of Fit, such as the loading bar.
	solver := G.NewAdamSolver(G.WithLearnRate(0.01))
	for epoch := 0; epoch <= 1000; epoch++ {
		loss, _ := modelFull.FitBatch(K.NamedTs{"x": x}, K.NamedTs{"yt": y}, solver)
		if epoch%100 == 0 {
			fmt.Printf("Epoch: %-4v Loss %.4f\n", fmt.Sprint(epoch), loss)
		}
	}

	/* OUTPUT
	Epoch: 0    Loss 0.2513
	Epoch: 100  Loss 0.2121
	Epoch: 200  Loss 0.0514
	Epoch: 300  Loss 0.0134
	Epoch: 400  Loss 0.0061
	Epoch: 500  Loss 0.0036
	Epoch: 600  Loss 0.0024
	Epoch: 700  Loss 0.0017
	Epoch: 800  Loss 0.0013
	Epoch: 900  Loss 0.0010
	Epoch: 1000 Loss 0.0008
	*/

	// And lets now see what the intermediate layer looks like once the whole model has trained.
	// This is different to the first time this is run, as the model has updated during training of the main model
	// Interestingly, we can see that a lot of the values are either very high or very low.
	TestModel(modelPartial, x, nil, "partial after training")

	/* OUTPUT
	Predictions (partial after training):
	X=[0  0] YP=[0.110  0.044  0.006  0.005  0.876]
	X=[0  1] YP=[0.887  0.760  0.287  0.285  0.103]
	X=[1  0] YP=[0.968  0.001  0.081  0.080  0.041]
	X=[1  1] YP=[0.999  0.059  0.865  0.868  0.001]
	*/

	// And lets just check the full model actually still performs XOR
	TestModel(modelFull, x, y, "full after training")

	/* OUTPUT
	Predictions (full after training):
	X=[0  0] Y=0 YP=0.029
	X=[0  1] Y=1 YP=0.979
	X=[1  0] Y=1 YP=0.965
	X=[1  1] Y=0 YP=0.029
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

func encoder(model *K.Model, inputs *G.Node) *G.Node {
	// As we are using two networks with different topologies, let's give differnt sections different names.
	// This is mandatory, as it allows a half network to always have the same layers have the same names as a full network.
	// i.e. the first encoder block will always be encoder_1, the first decoder will be decoder_1.
	n := K.NewNamer("encoder")
	outputs := K.Dense(model, n(), 5).MustAttach(inputs)
	outputs = K.Activation(model, n(), "sigmoid").MustAttach(outputs)
	return outputs
}

func decoder(model *K.Model, inputs *G.Node) *G.Node {
	n := K.NewNamer("decoder")
	outputs := K.Dense(model, n(), 1).MustAttach(inputs)
	outputs = K.Activation(model, n(), "sigmoid").MustAttach(outputs)
	return outputs
}

// Function to create a new model with randomized weights
func MakeModel(isJustForEncoding bool) *K.Model {
	// Create the empty model and a namer to provide names to the layers.
	// The namer is only used for the input layer
	model := K.NewModel(T.Float64)
	n := K.NewNamer("model")

	// Create the input layer
	inputs := K.Input(model, n(), 4, 2).Node()
	// Add the encoder. This is the first layer
	outputs := encoder(model, inputs)
	// If we want to, also add the decoder. This is the second layer
	if !isJustForEncoding {
		outputs = decoder(model, outputs)
	}

	model.MustBuild(K.WithInput("x", inputs), K.WithOutput("y", outputs), K.WithLoss(K.MSELoss("yt", outputs)))

	return model
}

// Function to run a model and print some values to the terminal
func TestModel(model *K.Model, x, y T.Tensor, testName string) {
	yps, err := model.Predict(K.NamedTs{"x": x})
	if err != nil {
		panic(err)
	}
	yp := yps["y"]

	fmt.Printf("\nPredictions (%s):\n", testName)
	for i := 0; i < yp.Shape()[0]; i++ {
		sx, _ := x.Slice(T.S(i))
		syp, _ := yp.Slice(T.S(i))
		if y != nil {
			sy, _ := y.Slice(T.S(i))
			fmt.Printf("X=%v Y=%v YP=%.3f\n", sx, sy, syp)
		} else {
			fmt.Printf("X=%v YP=%.3f\n", sx, syp)
		}

	}
	fmt.Println()
}
