package goras

import (
	"testing"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func makeXORModel() *Model {
	batchSize := 4
	inputNodes, hiddenNodes, outputNodes := 2, 5, 1

	model := NewModel(T.Float64)

	n := NewNamer("model")

	inputs := Input(model, n(), batchSize, inputNodes).Node()
	outputs := Dense(model, n(), hiddenNodes).MustAttach(inputs)
	outputs = Activation(model, n(), "sigmoid").MustAttach(outputs)
	outputs = Dense(model, n(), outputNodes).MustAttach(outputs)
	outputs = Sigmoid(model, n()).MustAttach(outputs)

	model.MustBuild(WithInput("x", inputs), WithOutput("yp", outputs), WithLoss(MSELoss("yt", outputs)))

	return model
}

func loadXORXY() (*T.Dense, *T.Dense) {
	x := T.New(
		T.WithShape(4, 2),
		T.WithBacking([]float64{0, 0, 0, 1, 1, 0, 1, 1}),
	)
	y := T.New(
		T.WithShape(4, 1),
		T.WithBacking([]float64{0, 1, 1, 0}),
	)
	return x, y
}
func TestXor(t *testing.T) {
	model := makeXORModel()
	solver := G.NewAdamSolver(G.WithLearnRate(0.01))
	x, y := loadXORXY()
	err := model.Fit(NamedTs{"x": x}, NamedTs{"yt": y}, solver, WithEpochs(1000), WithVerbose(false))
	if err != nil {
		t.Fatal(err)
	}
	yps, err := model.Predict(NamedTs{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	yp := yps["yp"]
	t.Log(yp)
	if yp.Shape()[0] != 4 || yp.Shape()[1] != 1 {
		t.Fatal("wrong output shape: ", yp.Shape())
	}
	a, _ := yp.At(0, 0)
	b, _ := yp.At(1, 0)
	c, _ := yp.At(2, 0)
	d, _ := yp.At(3, 0)
	if a.(float64) > 0.1 || b.(float64) < 0.9 || c.(float64) < 0.9 || d.(float64) > 0.1 {
		t.Fatal("wrong output: ", yp)
	}
}
