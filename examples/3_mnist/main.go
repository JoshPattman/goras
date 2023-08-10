package main

import (
	K "github.com/JoshPattman/goras"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func main() {
	MakeModel()
}

// This uses much of the same code from https://gorgonia.org/tutorials/mnist/
// We are also going to create a network with the exact same shape

func MakeModel() *K.Model {
	batchSize := 16
	model := K.NewModel()
	n := K.NewNamer("model")

	// Input shape is (batch_size, channels(this is one for b&w), img_x, img_y)
	inputs := K.Input(model, n.Next(), batchSize, 1, 28, 28).Node

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
	// There is no reshape in goras yet, so we will use the gorgonia one
	b, c, h, w := outputs.Shape()[0], outputs.Shape()[1], outputs.Shape()[2], outputs.Shape()[3]
	outputs, err := G.Reshape(outputs, T.Shape{b, c * h * w})
	if err != nil {
		panic(err)
	}
	outputs = K.Dropout(model, n.Next(), 0.2).MustAttach(outputs)

	// Dense net
	outputs = K.Dense(model, n.Next(), 625).MustAttach(outputs)
	outputs = K.Activation(model, n.Next(), "relu").MustAttach(outputs)
	outputs = K.Dropout(model, n.Next(), 0.55).MustAttach(outputs)
	outputs = K.Dense(model, n.Next(), 10).MustAttach(outputs)
	outputs = K.Activation(model, n.Next(), "softmax").MustAttach(outputs)

	model.MustBuild(K.WithInputs(inputs), K.WithOutputs(outputs), K.WithLosses(K.CCE))

	return model
}

// 32, 64, 128
// 625
// 10
