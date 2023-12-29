package goras

import (
	"fmt"
	"testing"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func makeXORModel() (*Model, error) {
	batchSize := 4
	inputNodes, hiddenNodes, outputNodes := 2, 5, 1

	model := NewModel()

	n := NewNamer("model")

	inputs := Input(model, n(), T.Float64, batchSize, inputNodes).Node()
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

	model := NewModel()

	n := NewNamer("model")

	inputs := Input(model, n(), T.Float64, batchSize, inputNodes).Node()
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

func TestTensorUtils(t *testing.T) {
	x := T.New(
		T.WithShape(4, 2),
		T.WithBacking([]float64{0, 0, 0, 1, 1, 0, 1, 1}),
	)
	xt, err := Make2DSliceTensor([][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if eq := x.Eq(xt); !eq {
		t.Fatal("Conv2DSliceToTensor failed: tensors not equal")
	}

	x32 := T.New(
		T.WithShape(4, 2),
		T.WithBacking([]float32{0, 0, 0, 1, 1, 0, 1, 1}),
	)
	xt32, err := Make2DSliceTensor([][]float32{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if eq := x32.Eq(xt32); !eq {
		t.Fatal("Conv2DSliceToTensor failed: tensors not equal")
	}

	y := T.New(
		T.WithShape(8, 1),
		T.WithBacking([]float64{0, 0, 0, 1, 1, 0, 1, 1}),
	)
	yt, err := Make1DSliceTensor([]float64{0, 0, 0, 1, 1, 0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if eq := y.Eq(yt); !eq {
		t.Fatal("Conv1DSliceToTensor failed: tensors not equal")
	}

	y32 := T.New(
		T.WithShape(8, 1),
		T.WithBacking([]float32{0, 0, 0, 1, 1, 0, 1, 1}),
	)
	yt32, err := Make1DSliceTensor([]float32{0, 0, 0, 1, 1, 0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if eq := y32.Eq(yt32); !eq {
		t.Fatal("Conv1DSliceToTensor failed: tensors not equal")
	}
}

func makeSingleActivationModel(ac func(*Model, string) *ActivationLayer, typ T.Dtype) (*Model, error) {
	model := NewModel()
	namer := NewNamer("model")
	inputs := Input(model, namer(), typ, 2, 3).Node()
	outputs, err := ac(model, namer()).Attach(inputs)
	if err != nil {
		return nil, err
	}
	err = model.Build(WithInput("x", inputs), WithOutput("yp", outputs), WithLoss(MSELoss("yt", outputs)))
	if err != nil {
		return nil, err
	}
	return model, nil
}

// Sum the element wise diff betweeen two 2D tensors
func epsilon2D(t1, t2 T.Tensor) (float32, error) {
	t, err := T.Sub(t1, t2)
	if err != nil {
		return 0, err
	}
	t, err = T.Abs(t)
	if err != nil {
		return 0, err
	}
	t, err = T.Sum(t, 0, 1)
	if err != nil {
		return 0, err
	}
	return t.Data().(float32), nil
}

func testActivation(ac func(*Model, string) *ActivationLayer, inp, tar T.Tensor) error {
	sigModel, err := makeSingleActivationModel(ac, T.Float32)
	if err != nil {
		return err
	}
	yp, err := sigModel.Predict(NamedTs{"x": inp})
	if err != nil {
		return err
	}
	if yp["yp"].Shape()[0] != 2 || yp["yp"].Shape()[1] != 3 {
		return fmt.Errorf("wrong output shape: %v", yp["yp"].Shape())
	}
	if ep, err := epsilon2D(yp["yp"], tar); err != nil || ep > 0.001 {
		return fmt.Errorf("wrong output. we inputted:\n%v\nwe got\n%v\nbut should have been\n%v", inp, yp["yp"], tar)
	}
	return nil
}

func TestActivations(t *testing.T) {
	x, err := Make2DSliceTensor(
		[][]float32{
			{-0.1, 0.3, 0.5},
			{-2, 1, 2},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	sigY, _ := Make2DSliceTensor(
		[][]float32{
			{0.47502081252106, 0.574442516811659, 0.622459331201854},
			{0.119202922022118, 0.731058578630005, 0.880797077977882},
		},
	)
	if err := testActivation(Sigmoid, x, sigY); err != nil {
		t.Fatal(err)
	}

	reluY, _ := Make2DSliceTensor(
		[][]float32{
			{0, 0.3, 0.5},
			{0, 1, 2},
		},
	)
	if err := testActivation(Relu, x, reluY); err != nil {
		t.Fatal(err)
	}

	tanhY, _ := Make2DSliceTensor(
		[][]float32{
			{-0.09966799462495582, 0.2913126124515909, 0.46211715726000974},
			{-0.9640275800758169, 0.7615941559557649, 0.9640275800758169},
		},
	)
	if err := testActivation(Tanh, x, tanhY); err != nil {
		t.Fatal(err)
	}

	binaryY, _ := Make2DSliceTensor(
		[][]float32{
			{0, 1, 1},
			{0, 1, 1},
		},
	)
	if err := testActivation(Binary, x, binaryY); err != nil {
		t.Fatal(err)
	}
	/*
		lr001 := func(m *Model, name string) *ActivationLayer {
			return LeakyRelu(m, name, 0.01)
		}
		lr001Y, _ := Make2DSliceTensor(
			[][]float32{
				{-0.001, 0.3, 0.5},
				{-0.02, 1, 2},
			},
		)
		if err := testActivation(lr001, x, lr001Y); err != nil {
			t.Fatal(err)
		}*/
}

func TestOneHot(t *testing.T) {
	model := NewModel()
	namer := NewNamer("model")
	inputs := Input(model, namer(), T.Int, 8).Node()
	outputs, err := OneHot(model, namer(), T.Float64, 5).Attach(inputs)
	if err != nil {
		t.Fatal(err)
	}
	err = model.Build(WithInput("x", inputs), WithOutput("yp", outputs), WithLoss(MSELoss("yt", outputs)))
	if err != nil {
		t.Fatal(err)
	}
	yp, err := model.Predict(NamedTs{"x": T.New(
		T.WithShape(8),
		T.WithBacking([]int{1, 3, 2, 0, 4, 1, 3, 2}),
	)})
	if err != nil {
		t.Fatal(err)
	}
	if yp["yp"].Shape()[0] != 8 || yp["yp"].Shape()[1] != 5 {
		t.Fatal("wrong output shape: ", yp["yp"].Shape())
	}
	if !yp["yp"].Eq(T.New(
		T.WithShape(8, 5),
		T.WithBacking([]float64{
			0, 1, 0, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 1, 0, 0,
			1, 0, 0, 0, 0,
			0, 0, 0, 0, 1,
			0, 1, 0, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 1, 0, 0,
		}),
	)) {
		t.Fatal("wrong output: ", yp["yp"])
	}

}
