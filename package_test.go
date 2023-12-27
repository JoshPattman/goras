package goras

import (
	"testing"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func makeXORModel() (*Model, error) {
	batchSize := 4
	inputNodes, hiddenNodes, outputNodes := 2, 5, 1

	model := NewModel(T.Float64)

	n := NewNamer("model")

	inputs := Input(model, n(), batchSize, inputNodes).Node()
	outputs, err := Dense(model, n(), hiddenNodes).Attach(inputs)
	if err != nil {
		return nil, err
	}
	outputs, err = Activation(model, n(), "sigmoid").Attach(outputs)
	if err != nil {
		return nil, err
	}
	outputs, err = Dense(model, n(), outputNodes).Attach(outputs)
	if err != nil {
		return nil, err
	}
	outputs, err = Sigmoid(model, n()).Attach(outputs)
	if err != nil {
		return nil, err
	}

	err = model.Build(WithInput("x", inputs), WithOutput("yp", outputs), WithLoss(MSELoss("yt", outputs)))
	if err != nil {
		return nil, err
	}
	return model, nil
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

// TEST: Creates, trains, and tests a model that learns the XOR function. If the model incorectly predicts, this test will fail (very unlikely).
func TestXor(t *testing.T) {
	model, err := makeXORModel()
	if err != nil {
		t.Fatal(err)
	}
	solver := G.NewAdamSolver(G.WithLearnRate(0.01))
	x, y := loadXORXY()
	err = model.Fit(NamedTs{"x": x}, NamedTs{"yt": y}, solver, WithEpochs(1000), WithVerbose(false))
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

// Dosent build the model
func makeUnfinishedXORModel() (*Model, *G.Node, *G.Node, error) {
	batchSize := 4
	inputNodes, hiddenNodes, outputNodes := 2, 5, 1

	model := NewModel(T.Float64)

	n := NewNamer("model")

	inputs := Input(model, n(), batchSize, inputNodes).Node()
	outputs, err := Dense(model, n(), hiddenNodes).Attach(inputs)
	if err != nil {
		return nil, nil, nil, err
	}
	outputs, err = Activation(model, n(), "sigmoid").Attach(outputs)
	if err != nil {
		return nil, nil, nil, err
	}
	outputs, err = Dense(model, n(), outputNodes).Attach(outputs)
	if err != nil {
		return nil, nil, nil, err
	}
	outputs, err = Sigmoid(model, n()).Attach(outputs)
	if err != nil {
		return nil, nil, nil, err
	}

	return model, inputs, outputs, nil
}
func TestMSELoss(t *testing.T) {
	model, inputs, outputs, err := makeUnfinishedXORModel()
	if err != nil {
		t.Fatal(err)
	}
	err = model.Build(WithInput("x", inputs), WithOutput("yp", outputs), WithLoss(MSELoss("yt", outputs)))
	if err != nil {
		t.Fatal(err)
	}
	solver := G.NewAdamSolver(G.WithLearnRate(0.01))
	x, y := loadXORXY()
	err = model.Fit(NamedTs{"x": x}, NamedTs{"yt": y}, solver, WithEpochs(1), WithVerbose(false))
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Predict(NamedTs{"x": x})
	if err != nil {
		t.Fatal(err)
	}
}

func TestBCELoss(t *testing.T) {
	model, inputs, outputs, err := makeUnfinishedXORModel()
	if err != nil {
		t.Fatal(err)
	}
	err = model.Build(WithInput("x", inputs), WithOutput("yp", outputs), WithLoss(BCELoss("yt", outputs)))
	if err != nil {
		t.Fatal(err)
	}
	solver := G.NewAdamSolver(G.WithLearnRate(0.01))
	x, y := loadXORXY()
	err = model.Fit(NamedTs{"x": x}, NamedTs{"yt": y}, solver, WithEpochs(1), WithVerbose(false))
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Predict(NamedTs{"x": x})
	if err != nil {
		t.Fatal(err)
	}
}

func TestCCELoss(t *testing.T) {
	model, inputs, outputs, err := makeUnfinishedXORModel()
	if err != nil {
		t.Fatal(err)
	}
	err = model.Build(WithInput("x", inputs), WithOutput("yp", outputs), WithLoss(CCELoss("yt", outputs)))
	if err != nil {
		t.Fatal(err)
	}
	solver := G.NewAdamSolver(G.WithLearnRate(0.01))
	x, y := loadXORXY()
	err = model.Fit(NamedTs{"x": x}, NamedTs{"yt": y}, solver, WithEpochs(1), WithVerbose(false))
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Predict(NamedTs{"x": x})
	if err != nil {
		t.Fatal(err)
	}
}
