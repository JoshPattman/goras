// In this example, we will create a very simple GAN (generative adversarial networks) that will learn to generate points on a sin curve.
// You should be able to take the principles in this example and apply them to more complex problems, such as generating faces.
package main

import (
	"fmt"
	"math"
	"math/rand"

	K "github.com/JoshPattman/goras"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func main() {
	// As we will be performing seperate training on the full model and just the generator, we need 3 distinct models.
	// However, as per example 2, we have linked the parameter of all 3.
	full, gen, disc := MakeModels()

	fmt.Println(full.Summary())
	fmt.Println(gen.Summary())
	fmt.Println(disc.Summary())

	solverG := G.NewAdamSolver(G.WithLearnRate(0.01))
	solverD := G.NewAdamSolver(G.WithLearnRate(0.01))

	for i := 0; i < 1000; i++ {
		trainingStep(full, gen, disc, solverG, solverD)
	}
}

// As a GAN requires multiple models, we are going to have to write a custom training step.
// I might incorporate a GAN training step into Goras one day, but this will do for now.
func trainingStep(full, gen, disc *K.Model, solverD, solverG G.Solver) {
	// We will start by generating a batch of generated samples from the generator
	// To do this, we need to start by creating a random input tensor of the correct dims (batchSize, latentDim)
	latentSamples := randomTensor(T.Shape{batchSize, latentDim})
	// And now we will predict a batch
	generatedSamples, err := gen.PredictBatch(K.V(latentSamples))
	if err != nil {
		panic(err)
	}

	// No training has happened yet, but now we will train the discriminator to pick if a given sample is real or fake
	// Lets start by generating some fake samples
	realSamples := randomSinTensor(T.Shape{batchSize, 2})
	// And now lets concat the first half of the real and first half of fake to make a full batch
	slicedGen, _ := generatedSamples.Slice(T.S(0, batchSize/2))
	slicedReal, _ := realSamples.Slice(T.S(0, batchSize/2))
	// And lets concat them together
	discriminatorX, _ := T.Concat(0, slicedGen, slicedReal)
	// Finally, lets create the labels. We will use 0 for fake and 1 for real
	labelsData := make([]float64, batchSize)
	for i := range labelsData {
		if i < batchSize/2 {
			labelsData[i] = 0
		} else {
			labelsData[i] = 1
		}
	}
	discriminatorY := T.New(T.WithShape(batchSize, 1), T.WithBacking(labelsData))

	// Ok, now we have the discriminator data, lets fit it for a single batch
	discLoss, err := disc.FitBatch(K.V(discriminatorX), K.V(discriminatorY), solverD)
	if err != nil {
		panic(err)
	}

	// Now its time to train the generator.
	// We will do this by using the full Model. We first generate some more latent space vectors, the pass them thorugh the entire model.
	// We then set the target output to be 0 (fake) for each element in the batch, and train.
	generatorX := randomTensor(T.Shape{batchSize, latentDim})
	generatorYData := make([]float64, batchSize)
	generatorY := T.New(T.WithShape(batchSize, 1), T.WithBacking(generatorYData))
	// Now lets fit
	genLoss, err := full.FitBatch(K.V(generatorX), K.V(generatorY), solverG)
	if err != nil {
		panic(err)
	}

	fmt.Println(genLoss, discLoss)
}

// Generate a random vector with values 0-1
func randomTensor(shape T.Shape) T.Tensor {
	data := make([]float64, shape.TotalSize())
	for i := range data {
		data[i] = rand.Float64()
	}
	tens := T.Tensor(T.New(T.WithShape(shape...), T.WithBacking(data)))
	return tens
}

// Generate a random vector with values 0-1
func randomSinTensor(shape T.Shape) T.Tensor {
	data := make([]float64, shape.TotalSize())
	for i := range data {
		if i%2 != 0 {
			// This is a y pos. We also want to make sure its between 0 and 1
			data[i] = (math.Sin(data[i-1]) + 1) / 2
		} else {
			// This is an x pos
			data[i] = rand.Float64()
		}
	}
	tens := T.Tensor(T.New(T.WithShape(batchSize, 2), T.WithBacking(data)))
	return tens
}
