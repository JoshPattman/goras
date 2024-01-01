// This example uses a data generator to generate data on-the-fly during training.
// We fit a simple model to a sing function.
// We also then use gonum/plot to plot the results, allowing us to visualise the function.
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/JoshPattman/goras"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	// Create a data generator. This creates data on-the-fly instead of loading it all into memory.
	trainingGenerator := NewSinDataGenerator("x", "yt", 5000)

	// Create a model. We will use a simple dense model with 2 hidden layers, each with 10 neurons.
	// We will use the relu activation function for the hidden layers, and the tanh activation function for the output layer.
	batchSize := 16
	model := goras.NewModel()
	namer := goras.NewNamer("model")
	inp := goras.Input(model, namer(), tensor.Float64, batchSize, 1).Node()
	out := goras.Dense(model, namer(), 10).MustAttach(inp)
	out = goras.Relu(model, namer()).MustAttach(out)
	out = goras.Dense(model, namer(), 10).MustAttach(out)
	out = goras.Relu(model, namer()).MustAttach(out)
	out = goras.Dense(model, namer(), 1).MustAttach(out)
	out = goras.Tanh(model, namer()).MustAttach(out)
	model.MustBuild(goras.WithInput("x", inp), goras.WithOutput("yp", out), goras.WithLoss(goras.MSELoss("yt", out)))

	// Create a solver and fit the model.
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.01))
	if err := model.FitGenerator(trainingGenerator, solver, goras.WithEpochs(10)); err != nil {
		panic(err)
	}

	// Create some test X and Y data
	testX := tensor.NewDense(tensor.Float64, tensor.Shape{360, 1})
	testY := tensor.NewDense(tensor.Float64, tensor.Shape{360, 1})
	for i := 0.0; i < 360; i++ {
		x := i * 2 * 3.14159 / 360
		MustSetAt(testX, x, int(i), 0)
		MustSetAt(testY, math.Sin(x), int(i), 0)
	}

	// Predict the test data. Hopefully, this should be close to the real data.
	testYPs, err := model.Predict(goras.NamedTs{
		"x": testX,
	})
	if err != nil {
		panic(err)
	}
	testYP := testYPs["yp"]

	// Create a plot to visualise the results
	p := plot.New()
	p.X.Label.Text = "X (rads)"
	p.Y.Label.Text = "Y"

	// Build the data for the plot
	realData := plotter.XYs{}
	predictedData := plotter.XYs{}
	for i := 0; i < testY.Shape()[0]; i++ {
		realData = append(realData, plotter.XY{X: MustAt(testX, i, 0).(float64), Y: MustAt(testY, i, 0).(float64)})
		predictedData = append(predictedData, plotter.XY{X: MustAt(testX, i, 0).(float64), Y: MustAt(testYP, i, 0).(float64)})
	}

	// Add the lines to the plot
	realLine, err := plotter.NewLine(realData)
	if err != nil {
		panic(err)
	}
	realLine.Color = plotutil.Color(0)
	predLine, err := plotter.NewLine(predictedData)
	if err != nil {
		panic(err)
	}
	predLine.Color = plotutil.Color(1)
	p.Add(realLine, predLine, plotter.NewGrid())

	// Save the plot to a file
	if err := p.Save(512, 512, "sin.png"); err != nil {
		panic(err)
	}

}

// MustAt is a helper function to get a value from a tensor and panic if there is an error.
func MustAt(t tensor.Tensor, i ...int) interface{} {
	v, err := t.At(i...)
	if err != nil {
		panic(err)
	}
	return v
}

// MustSetAt is a helper function to set a value in a tensor and panic if there is an error.
func MustSetAt(t tensor.Tensor, v interface{}, i ...int) {
	if err := t.SetAt(v, i...); err != nil {
		panic(err)
	}
}

// Ensure SinDataGenerator implements goras.TrainingDataGenerator.
var _ goras.TrainingDataGenerator = &SinDataGenerator{}

// NewSinDataGenerator creates a new SinDataGenerator.
func NewSinDataGenerator(inputName, targetOutputName string, numSamplesPerEpoch int) *SinDataGenerator {
	return &SinDataGenerator{
		inputName:          inputName,
		targetOutputName:   targetOutputName,
		numSamplesPerEpoch: numSamplesPerEpoch,
		batchSize:          -1,
		batchesLeft:        -1,
		batchesForEpoch:    -1,
	}
}

// SinDataGenerator generates training data for a sin function.
// A data generator is used by a model to generate data on-the-fly during training.
// You would usually use a data generator when you have a large dataset that cannot fit into memory,
// or when you want to generate data on-the-fly, but for now we'll just use it to generate a sin function.
type SinDataGenerator struct {
	inputName          string
	targetOutputName   string
	numSamplesPerEpoch int
	batchSize          int
	batchesLeft        int
	batchesForEpoch    int
}

// NextBatch implements goras.TrainingDataGenerator.
func (dg *SinDataGenerator) NextBatch() (map[string]tensor.Tensor, map[string]tensor.Tensor, error) {
	if dg.batchSize == -1 {
		return nil, nil, fmt.Errorf("did not call Reset() on SinDataGenerator")
	} else if dg.batchesLeft == 0 {
		return nil, nil, nil
	}
	dataX := make([][]float64, 0)
	dataY := make([][]float64, 0)
	for i := 0; i < dg.batchSize; i++ {
		x := rand.Float64() * 2 * 3.14159
		dataX = append(dataX, []float64{x})
		dataY = append(dataY, []float64{math.Sin(x)})
	}
	xT, err := goras.Make2DSliceTensor(dataX)
	if err != nil {
		return nil, nil, err
	}
	yT, err := goras.Make2DSliceTensor(dataY)
	if err != nil {
		return nil, nil, err
	}
	dg.batchesLeft--
	return goras.NamedTs{
			dg.inputName: xT,
		}, goras.NamedTs{
			dg.targetOutputName: yT,
		}, nil
}

// NumBatches implements goras.TrainingDataGenerator.
func (dg *SinDataGenerator) NumBatches() int {
	return dg.batchesForEpoch
}

// Reset implements goras.TrainingDataGenerator.
func (dg *SinDataGenerator) Reset(batchSize int) error {
	dg.batchSize = batchSize
	dg.batchesLeft = dg.numSamplesPerEpoch / batchSize
	dg.batchesForEpoch = dg.batchesLeft
	return nil
}
