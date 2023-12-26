package main

import (
	K "github.com/JoshPattman/goras"
	T "gorgonia.org/tensor"
)

// Function to make a dogs/cats classifier
func MakeModel() *K.Model {
	m := K.NewModel(T.Float64)
	n := K.NewNamer("model")

	// The shape is (batch, channels (3 because the images are RGB), height, width)
	input := K.Input(m, n(), 32, 3, 64, 64).Node()

	// Current Shape: (32,3,64,64)
	output := K.SimpleConv2D(m, n(), 3, 16).MustAttach(input)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.SimpleMaxPooling2D(m, n(), 2).MustAttach(output)
	// Current Shape: (32,16,32,32)
	output = K.SimpleConv2D(m, n(), 3, 32).MustAttach(output)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.SimpleMaxPooling2D(m, n(), 2).MustAttach(output)
	// Current Shape: (32,32,16,16)
	output = K.SimpleConv2D(m, n(), 3, 64).MustAttach(output)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.SimpleMaxPooling2D(m, n(), 2).MustAttach(output)
	// Current Shape: (32,64,8,8)
	output = K.Reshape(m, n(), T.Shape{32, 64 * 8 * 8}).MustAttach(output)
	// Current Shape: (32,4096)
	output = K.Dense(m, n(), 128).MustAttach(output)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.Dense(m, n(), 64).MustAttach(output)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.Dense(m, n(), 1).MustAttach(output)
	output = K.Sigmoid(m, n()).MustAttach(output)

	// We are using BCE (binary cross-entropy) loss as it is better for binary classifiers than MSE
	m.MustBuild(K.WithInput("", input), K.WithOutput("y", output), K.WithLoss(K.BCELoss("yt", output)))

	return m
}
