package main

import (
	K "github.com/JoshPattman/goras"
	T "gorgonia.org/tensor"
)

func MakeModel() *K.Model {
	m := K.NewModel(T.Float64)
	n := K.NewNamer("model")

	input := K.Input(m, n.Next(), 32, 3, 64, 64).Node

	// (32,3,64,64)
	output := K.SimpleConv2D(m, n.Next(), 3, 16).MustAttach(input)
	output = K.Relu(m, n.Next()).MustAttach(output)
	output = K.SimpleMaxPooling2D(m, n.Next(), 2).MustAttach(output)
	// (32,16,32,32)
	output = K.SimpleConv2D(m, n.Next(), 3, 32).MustAttach(output)
	output = K.Relu(m, n.Next()).MustAttach(output)
	output = K.SimpleMaxPooling2D(m, n.Next(), 2).MustAttach(output)
	// (32,32,16,16)
	output = K.SimpleConv2D(m, n.Next(), 3, 64).MustAttach(output)
	output = K.Relu(m, n.Next()).MustAttach(output)
	output = K.SimpleMaxPooling2D(m, n.Next(), 2).MustAttach(output)
	// (32,64,8,8)
	output = K.Reshape(m, n.Next(), T.Shape{32, 64 * 8 * 8}).MustAttach(output)
	output = K.Dense(m, n.Next(), 128).MustAttach(output)
	output = K.Relu(m, n.Next()).MustAttach(output)
	output = K.Dense(m, n.Next(), 64).MustAttach(output)
	output = K.Relu(m, n.Next()).MustAttach(output)
	output = K.Dense(m, n.Next(), 1).MustAttach(output)
	output = K.Sigmoid(m, n.Next()).MustAttach(output)

	m.MustBuild(K.WithInputs(input), K.WithOutputs(output), K.WithLosses(K.BCE))

	return m
}
