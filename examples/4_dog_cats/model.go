package main

import (
	K "github.com/JoshPattman/goras"
	T "gorgonia.org/tensor"
)

func MakeModel() *K.Model {
	m := K.NewModel(T.Float64)
	n := K.NewNamer("model")

	input := K.Input(m, n(), 32, 3, 64, 64).Node()

	// (32,3,64,64)
	output := K.SimpleConv2D(m, n(), 3, 16).MustAttach(input)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.SimpleMaxPooling2D(m, n(), 2).MustAttach(output)
	// (32,16,32,32)
	output = K.SimpleConv2D(m, n(), 3, 32).MustAttach(output)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.SimpleMaxPooling2D(m, n(), 2).MustAttach(output)
	// (32,32,16,16)
	output = K.SimpleConv2D(m, n(), 3, 64).MustAttach(output)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.SimpleMaxPooling2D(m, n(), 2).MustAttach(output)
	// (32,64,8,8)
	output = K.Reshape(m, n(), T.Shape{32, 64 * 8 * 8}).MustAttach(output)
	output = K.Dense(m, n(), 128).MustAttach(output)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.Dense(m, n(), 64).MustAttach(output)
	output = K.Relu(m, n()).MustAttach(output)
	output = K.Dense(m, n(), 1).MustAttach(output)
	output = K.Sigmoid(m, n()).MustAttach(output)

	m.MustBuild(K.WithInputs(input), K.WithOutputs(output), K.WithLosses(K.BCE))

	return m
}
