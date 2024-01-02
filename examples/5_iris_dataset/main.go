// This file demonstrates how to do regression on the iris dataset using goras.
// It does not explain stuff as much as the earlier tutorials, so it's probably best to read them first.
package main

import (
	"fmt"
	"os"

	"github.com/JoshPattman/datautil"
	"github.com/JoshPattman/goras"
	"github.com/go-gota/gota/dataframe"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Loads the X and Y from the iris dataset.
// Does NOT perform normalisation on the inputs. You should probably do this with anything bigger than a linear regression model.
func loadDataset() (*tensor.Dense, *tensor.Dense) {
	// Read the csv with gota
	f, _ := os.Open("iris.csv")
	defer f.Close()
	df := dataframe.ReadCSV(f)
	// Split the Label column into 3 columns (one hot encoded): Label:Iris-virginica, Label:Iris-setosa, Label:Iris-versicolor
	df = datautil.SplitCategoricalColumn(df, "Label", []string{"Iris-virginica", "Iris-setosa", "Iris-versicolor"})
	// Extract the X and Y from the dataframe and convert to tensors
	X := datautil.ColumnsToTensor(df, []string{"SepalLength", "SepalWidth", "PetalLength", "PetalWidth"})
	Y := datautil.ColumnsToTensor(df, []string{"Label:Iris-virginica", "Label:Iris-setosa", "Label:Iris-versicolor"})
	return X, Y
}

func main() {
	// Load the dataset and print some info
	X, Y := loadDataset()
	fmt.Println(X)
	fmt.Println(Y)
	fmt.Println(X.Shape(), Y.Shape())

	// Create the model. As it is a linear regression model, we only need a single dense layer.
	// We also use a batch size of 16.
	model := goras.NewModel()
	inp := goras.Input(model, "input", tensor.Float64, 16, 4).Node()
	out := goras.Dense(model, "dense", 3).MustAttach(inp)
	out = goras.Softmax(model, "activation").MustAttach(out)

	// Build the model with CCE loss.
	model.MustBuild(goras.WithInput("x", inp), goras.WithOutput("y", out), goras.WithLoss(goras.CCELoss("yt", out)))

	// Print a summary of the model
	fmt.Println(model.Summary())

	// Set up a callback to test the accuracy of the model
	accuracyCallback := func() (float64, error) {
		acc := calculateAccuracy(model, X, Y)
		return acc, nil
	}

	// Create a file to log the metrics to
	logFile, err := os.Create("metrics.csv")
	if err != nil {
		panic(err)
	}
	defer logFile.Close()

	// Fit the model with an Adam solver and 500 epochs. Also log the metrics to a csv file.
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.02))
	model.MustFit(
		goras.NamedTs{"x": X}, goras.NamedTs{"yt": Y},
		solver,
		goras.WithEpochs(500),
		goras.WithLoggingEvery(50),
		goras.WithEpochCallbacks(
			goras.CustomMetricCallback(accuracyCallback, "accuracy", 1),
			goras.LogCSVMetricsCallback(logFile, "loss", "accuracy"),
		),
	)

	// Print the predictions of the model (this is measured on the training set, it would be better practice to do this on an unseen test set)
	py, _ := model.Predict(goras.NamedTs{"x": X})

	correct := 0
	for i := 0; i < 30; i++ {
		xi, _ := X.Slice(tensor.S(i))
		yi, _ := Y.Slice(tensor.S(i))
		pyi, _ := py["y"].Slice(tensor.S(i))
		classNames := []string{"\033[0;101mIris-virginica\033[0m", "\033[0;102mIris-setosa\033[0m", "\033[0;103mIris-versicolor\033[0m"}
		classI, _ := tensor.Argmax(pyi, 0)
		aclassI, _ := tensor.Argmax(yi, 0)
		class := classNames[classI.Data().(int)]
		aclass := classNames[aclassI.Data().(int)]
		fmt.Printf("| X: %.2f, Y: %v, YP: %.2f |\t| Actual Class: %-30s Predicted Class: %-30s |\n", xi, yi, pyi, aclass, class)
		if class == aclass {
			correct++
		}
	}
	fmt.Printf("Accuracy: %.2f%%\n", float64(correct)/30*100) // My model here is getting 96.67%
}

func calculateAccuracy(model *goras.Model, X, Y *tensor.Dense) float64 {
	YPs, _ := model.Predict(goras.NamedTs{"x": X})
	correct := 0
	for i := 0; i < 30; i++ {
		yi, _ := Y.Slice(tensor.S(i))
		pyi, _ := YPs["y"].Slice(tensor.S(i))
		classNames := []string{"\033[0;101mIris-virginica\033[0m", "\033[0;102mIris-setosa\033[0m", "\033[0;103mIris-versicolor\033[0m"}
		classI, _ := tensor.Argmax(pyi, 0)
		aclassI, _ := tensor.Argmax(yi, 0)
		class := classNames[classI.Data().(int)]
		aclass := classNames[aclassI.Data().(int)]
		if class == aclass {
			correct++
		}
	}
	return float64(correct) / 30 * 100
}
