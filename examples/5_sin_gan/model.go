package main

import (
	"strings"

	K "github.com/JoshPattman/goras"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

var (
	batchSize int = 8
	latentDim int = 5
)

func MakeModels() (full, generator, discriminator *K.Model) {
	fullModel := K.NewModel(T.Float64)
	genModel := K.NewModel(T.Float64)
	discModel := K.NewModel(T.Float64)

	fullInputs := K.Input(fullModel, "full_input", batchSize, latentDim).Node()
	genInputs := K.Input(genModel, "gen_input", batchSize, latentDim).Node()
	discInputs := K.Input(discModel, "disc_input", batchSize, 2).Node() // Only inputs dim of 2 as there is only x and y pos of sin wave

	fullOutputs := attachDiscriminator(fullModel, attachGenerator(fullModel, fullInputs), false) // The discrim layers should not be trainable in the full model
	genOutputs := attachGenerator(genModel, genInputs)
	discOutputs := attachDiscriminator(discModel, discInputs, true)

	fullModel.MustBuild(K.WithInputs(fullInputs), K.WithOutputs(fullOutputs), K.WithLosses(K.BCE))
	genModel.MustBuild(K.WithInputs(genInputs), K.WithOutputs(genOutputs), K.WithLosses(K.BCE))
	discModel.MustBuild(K.WithInputs(discInputs), K.WithOutputs(discOutputs), K.WithLosses(K.BCE))

	genModel.BindParamsFrom(fullModel)
	discModel.BindParamsFrom(fullModel)

	return fullModel, genModel, discModel
}

func attachGenerator(model *K.Model, input *G.Node) *G.Node {
	n := K.NewNamer("generator")
	output := K.Dense(model, n(), 25).MustAttach(input)
	output = K.Sigmoid(model, n()).MustAttach(output)
	output = K.Dropout(model, n(), 0.2).MustAttach(output)
	output = K.Dense(model, n(), 25).MustAttach(output)
	output = K.Sigmoid(model, n()).MustAttach(output)
	output = K.Dropout(model, n(), 0.2).MustAttach(output)
	output = K.Dense(model, n(), 2).MustAttach(output)
	output = K.Sigmoid(model, n()).MustAttach(output)
	return output
}

func attachDiscriminator(model *K.Model, input *G.Node, isTrainable bool) *G.Node {
	n := K.NewNamer("discriminator")
	output := K.Dense(model, n(), 25).MustAttach(input)
	output = K.Sigmoid(model, n()).MustAttach(output)
	output = K.Dropout(model, n(), 0.2).MustAttach(output)
	output = K.Dense(model, n(), 25).MustAttach(output)
	output = K.Sigmoid(model, n()).MustAttach(output)
	output = K.Dropout(model, n(), 0.2).MustAttach(output)
	output = K.Dense(model, n(), 1).MustAttach(output)
	output = K.Sigmoid(model, n()).MustAttach(output)

	if !isTrainable {
		for _, l := range model.Layers {
			if strings.Contains(l.Name(), "discriminator") {
				l.SetTrainable(false)
			}
		}
	}
	return output
}
