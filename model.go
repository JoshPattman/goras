package goras

import (
	"encoding/gob"
	"fmt"
	"io"
	"strings"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// Model is the core primitive of goras.
// It is effectively a wrapper around a Gorgonia graph, with extra functionality.
type Model struct {
	Graph             *G.ExprGraph
	Layers            []Layer
	Machine           G.VM
	InputNodes        map[string]*G.Node
	OutputNodes       map[string]*G.Node
	OutputValues      map[string]*G.Value // This is deliberately a ref because i think maps are scary
	LossValue         G.Value
	LossRequiredNodes map[string]*G.Node
}

// NewModel creates a new model with no layers
func NewModel() *Model {
	return &Model{Graph: G.NewGraph(), Layers: []Layer{}}
}

// AddLayer adds a layer to the model. You usually don't need to call this directly, as the layer constructors do it for you.
func (m *Model) AddLayer(l Layer) {
	m.Layers = append(m.Layers, l)
}

type buildParams struct {
	inputNodes  map[string]*G.Node
	outputNodes map[string]*G.Node
	loss        LossFunc
}

// BuildOpts are options for the Build method.
type BuildOpts func(*buildParams)

// WithInput adds an input node to the model.
//   - inputName: The name we will use to pass tensors to this node. This must be unique, and will be used later in fit and predict methods.
//   - inputNode: The node to use as the input. This is usually from a goras.Input layer.
func WithInput(inputName string, inputNode *G.Node) BuildOpts {
	return func(b *buildParams) { b.inputNodes[inputName] = inputNode }
}

// WithOutput adds an output node to the model.
//   - outputName: The name we will use to get tensors from this node. This must be unique, and will be used later in fit and predict methods.
//   - outputNode: The node to use as the output.
func WithOutput(name string, outputNode *G.Node) BuildOpts {
	return func(b *buildParams) { b.outputNodes[name] = outputNode }
}

// WithLoss specifies the loss function for the model.
func WithLoss(loss LossFunc) BuildOpts {
	return func(b *buildParams) { b.loss = loss }
}

// Build builds the model, using a specified input and output node.
// It adds the loss function to the graph, and creates the machine.
// This should only be called once per model.
func (m *Model) Build(opts ...BuildOpts) error {
	buildParams := &buildParams{
		inputNodes:  make(map[string]*G.Node),
		outputNodes: make(map[string]*G.Node),
	}
	for _, opt := range opts {
		opt(buildParams)
	}
	if len(buildParams.inputNodes) == 0 || len(buildParams.outputNodes) == 0 {
		return fmt.Errorf("must at least have one input and output node")
	}
	if buildParams.loss == nil {
		return fmt.Errorf("loss must be specified")
	}

	// Store input and output nodes
	m.InputNodes = buildParams.inputNodes
	m.OutputNodes = buildParams.outputNodes
	// Read the outputs to values
	m.OutputValues = make(map[string]*G.Value, len(m.OutputNodes))
	for name := range m.OutputNodes {
		var val G.Value
		G.Read(m.OutputNodes[name], &val)
		m.OutputValues[name] = &val
	}
	// Define loss function
	lossNode, lossRequiredNodes, err := buildParams.loss()
	if err != nil {
		return fmt.Errorf("error while adding loss: %v", err)
	}
	G.Read(lossNode, &m.LossValue)
	m.LossRequiredNodes = lossRequiredNodes
	trainables := m.Trainables()
	if len(trainables) != 0 {
		_, err = G.Grad(lossNode, trainables...)
		if err != nil {
			return fmt.Errorf("error while computing grad: %v", err)
		}
	}

	// Check for duplicate node names
	nodeNames := make(map[string]bool)
	for _, n := range m.Graph.AllNodes() {
		if _, ok := nodeNames[n.Name()]; ok {
			return fmt.Errorf("duplicate node name %s, either there are two layers with the same name, or this is a bug (please report)", n.Name())
		}
		nodeNames[n.Name()] = true
	}

	// Check for duplicate layer names
	layerNames := make(map[string]bool)
	for _, l := range m.Layers {
		if _, ok := layerNames[l.Name()]; ok {
			return fmt.Errorf("duplicate layer name %s, either there are two layers with the same name, or this is a bug (please report)", l.Name())
		}
		layerNames[l.Name()] = true
	}

	// Create machine
	m.Machine = G.NewTapeMachine(m.Graph, G.BindDualValues(m.Trainables()...))
	return nil
}

// MustBuild calls Build, but panics if there is an error.
func (m *Model) MustBuild(opts ...BuildOpts) {
	err := m.Build(opts...)
	if err != nil {
		panic(err)
	}
}

// Trainables returns a list of all the trainable nodes in the model.
func (m *Model) Trainables() G.Nodes {
	var ret G.Nodes
	for _, l := range m.Layers {
		if l.Trainable() {
			for _, t := range l.Parameters() {
				ret = append(ret, t)
			}
		}
	}
	return ret
}

// valueToTensor converts a G.Value to a tensor.
// The tensor shares the same underlying data as the value, so changing the returned tensor will change the value.
func valueToTensor(v G.Value) *T.Dense {
	return T.New(T.WithShape(v.Shape()...), T.WithBacking(v.Data()))
}

// GetParams returns a map of all the parameters in the model.
// The keys are the layer name and parameter name, separated by a colon (e.g. "model_1:weights")
func (m *Model) GetParams() map[string]*T.Dense {
	ret := make(map[string]*T.Dense)
	for _, l := range m.Layers {
		for k, v := range l.Parameters() {
			ret[l.Name()+":"+k] = valueToTensor(v.Value())
		}
	}
	return ret
}

// SetParams sets the parameters in the model, which can be retrieved with Model.GetParams.
// It will only load parameters with matching names, and will ignore any others.
// This means you can load parameters from a model with a different architecture, as long as the names match on equivalent layers.
func (m *Model) SetParams(params map[string]*T.Dense) error {
	for _, l := range m.Layers {
		for k, v := range l.Parameters() {
			if p, ok := params[l.Name()+":"+k]; ok {
				if err := G.Let(v, p); err != nil {
					return fmt.Errorf("error setting parameter %s: %s", l.Name()+":"+k, err)
				}
			}
		}
	}
	return nil
}

// MustSetParams calls SetParams, but panics if there is an error.
func (m *Model) MustSetParams(params map[string]*T.Dense) {
	err := m.SetParams(params)
	if err != nil {
		panic(err)
	}
}

// WriteParams writes the parameters in gob format to an io.Writer.
// The params are retrieved with Model.GetParams.
func (m *Model) WriteParams(w io.Writer) error {
	params := m.GetParams()
	enc := gob.NewEncoder(w)
	return enc.Encode(params)
}

// MustWriteParams calls WriteParams, but panics if there is an error.
func (m *Model) MustWriteParams(w io.Writer) {
	err := m.WriteParams(w)
	if err != nil {
		panic(err)
	}
}

// ReadParams reads the parameters in gob format from an io.Reader.
// The params are retrieved with Model.GetParams.
func (m *Model) ReadParams(r io.Reader) error {
	var params map[string]*T.Dense
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&params); err != nil {
		return err
	}
	return m.SetParams(params)
}

// MustReadParams calls ReadParams, but panics if there is an error.
func (m *Model) MustReadParams(r io.Reader) {
	err := m.ReadParams(r)
	if err != nil {
		panic(err)
	}
}

// BindParamsFrom binds the parameters in the model m1 to the parameters in this model m, meaning layers with the same name will share the same tensors.
// This is a bit of a hack to allow two models to train the same weights.
// This can be called multiple times, where later binds may override earlier ones.
// For example, if you are making an autoencoder, you would have one main model for training, and an encoder model and decoder model which are bound to that.
// That then allows you to run partial bits of the network.
func (m *Model) BindParamsFrom(m1 *Model) error {
	paramsSrc := m1.GetParams()
	for _, l := range m.Layers {
		for k, v := range l.Parameters() {
			if p, ok := paramsSrc[l.Name()+":"+k]; ok {
				if err := G.Let(v, p); err != nil {
					return fmt.Errorf("error binding parameter %s: %s", l.Name()+":"+k, err)
				}
			}
		}
	}
	return nil
}

// MustBindParamsFrom calls BindParamsFrom, but panics if there is an error.
func (m *Model) MustBindParamsFrom(m1 *Model) {
	err := m.BindParamsFrom(m1)
	if err != nil {
		panic(err)
	}
}

// CopyParamsFrom copys the parameters in the model m1 to the parameters in this model m, meaning layers with the same name will share the same values in their tensors.
// The tensors will be copies of each other, so changing one will not change the other.
// If you want to share the tensors, use BindParamsFrom instead.
func (m *Model) CopyParamsFrom(m1 *Model) error {
	paramsSrc := m1.GetParams()
	for _, l := range m.Layers {
		for k, v := range l.Parameters() {
			if p, ok := paramsSrc[l.Name()+":"+k]; ok {
				pCopy := p.Clone().(*T.Dense)
				if err := G.Let(v, pCopy); err != nil {
					return fmt.Errorf("error copying parameter %s: %s", l.Name()+":"+k, err)
				}
			}
		}
	}
	return nil
}

// MustCopyParamsFrom calls CopyParamsFrom, but panics if there is an error.
func (m *Model) MustCopyParamsFrom(m1 *Model) {
	err := m.CopyParamsFrom(m1)
	if err != nil {
		panic(err)
	}
}

// PredictBatch runs the model on a batch of input data. The batch size must match the input node shape.
func (m *Model) PredictBatch(inputs map[string]T.Tensor) (map[string]T.Tensor, error) {
	if err := checkBatchedInputShapes(m, inputs); err != nil {
		return nil, err
	}
	m.Machine.Reset()
	for name := range inputs {
		if err := G.Let(m.InputNodes[name], inputs[name]); err != nil {
			return nil, err
		}
	}
	// Set every loss required node to a tensor of the correct shape
	for _, n := range m.LossRequiredNodes {
		if err := G.Let(n, T.New(T.WithShape(n.Shape()...), T.Of(n.Dtype()))); err != nil {
			return nil, err
		}
	}
	// Run the machine
	if err := m.Machine.RunAll(); err != nil {
		return nil, err
	}
	// We need to clone here otherwise the next time the machine is run, the tensor will be changed
	outputTensors := make(map[string]T.Tensor, len(m.OutputNodes))
	for name := range m.OutputValues {
		outputTensors[name] = T.New(
			T.WithShape((*m.OutputValues[name]).Shape()...),
			T.WithBacking((*m.OutputValues[name]).Data()),
		).Clone().(*T.Dense)
	}
	return outputTensors, nil
}

// MustPredictBatch calls PredictBatch, but panics if there is an error.
func (m *Model) MustPredictBatch(inputs map[string]T.Tensor) map[string]T.Tensor {
	ys, err := m.PredictBatch(inputs)
	if err != nil {
		panic(err)
	}
	return ys
}

// FitBatch runs the model on a batch of input data, and then trains the model on the target data.
// The solver used is passed in as an argument.
// IMPORTANT NOTE: Currently, when the data is batched, the last batch of data will be discarded if the x size does not evenly divide the batch size.
func (m *Model) FitBatch(inputs, lossRequirements map[string]T.Tensor, solver G.Solver) (float64, error) {
	if err := checkBatchedInputShapes(m, inputs); err != nil {
		return 0, err
	}
	if err := checkBatchedLossRequirementShapes(m, lossRequirements); err != nil {
		return 0, err
	}
	m.Machine.Reset()
	for name := range inputs {
		if err := G.Let(m.InputNodes[name], inputs[name]); err != nil {
			return 0, err
		}
	}
	for name := range lossRequirements {
		if err := G.Let(m.LossRequiredNodes[name], lossRequirements[name]); err != nil {
			return 0, err
		}
	}
	if err := m.Machine.RunAll(); err != nil {
		return 0, err
	}
	if err := solver.Step(G.NodesToValueGrads(m.Trainables())); err != nil {
		return 0, err
	}
	loss := 0.0
	switch m.LossValue.Dtype() {
	case T.Float64:
		loss = m.LossValue.Data().(float64)
	case T.Float32:
		loss = float64(m.LossValue.Data().(float32))
	default:
		return 0, fmt.Errorf("unsupported loss dtype %v, please use either float64 or float32", m.LossValue.Dtype())
	}
	return loss, nil
}

// MustFitBatch calls FitBatch, but panics if there is an error.
func (m *Model) MustFitBatch(inputs, lossRequirements map[string]T.Tensor, solver G.Solver) float64 {
	loss, err := m.FitBatch(inputs, lossRequirements, solver)
	if err != nil {
		panic(err)
	}
	return loss
}

// FitOpts are options for the Fit method.
type FitOpt func(*fitParams)

type fitParams struct {
	Epochs            int
	LogEvery          int
	Verbose           bool
	ClearLine         bool
	EpochEndCallbakcs []EpochCallback
}

// WithEpochs sets the number of epochs to train for.
func WithEpochs(epochs int) FitOpt { return func(p *fitParams) { p.Epochs = epochs } }

// WithLoggingEvery sets how often to log the loss.
func WithLoggingEvery(epochs int) FitOpt { return func(p *fitParams) { p.LogEvery = epochs } }

// WithVerbose sets whether to log the loss.
func WithVerbose(verbose bool) FitOpt { return func(p *fitParams) { p.Verbose = verbose } }

// WithClearLine sets whether to clear the line when logging the loss.
func WithClearLine(clear bool) FitOpt { return func(p *fitParams) { p.ClearLine = clear } }

// WithEpochCallback adds a callback to be called at the end of each epoch.
func WithEpochCallback(cb EpochCallback) FitOpt {
	return func(p *fitParams) { p.EpochEndCallbakcs = append(p.EpochEndCallbakcs, cb) }
}

// Fit fits the model to the given data.
func (m *Model) Fit(xs, ys map[string]T.Tensor, solver G.Solver, opts ...FitOpt) error {
	return m.FitGenerator(NewTTDG(xs, ys), solver, opts...)
}

// MustFit calls Fit, but panics if there is an error.
func (m *Model) MustFit(xs, ys map[string]T.Tensor, solver G.Solver, opts ...FitOpt) {
	err := m.Fit(xs, ys, solver, opts...)
	if err != nil {
		panic(err)
	}
}

// FitGenerator fits the model to the given data generator.
func (m *Model) FitGenerator(tdg TrainingDataGenerator, solver G.Solver, opts ...FitOpt) error {
	params := &fitParams{
		Epochs:            1,
		LogEvery:          1,
		Verbose:           true,
		ClearLine:         false,
		EpochEndCallbakcs: []EpochCallback{},
	}
	for _, o := range opts {
		o(params)
	}
	batchSize := m.getCurrentBatchSize()
	for epoch := 1; epoch <= params.Epochs; epoch++ {
		tdg.Reset(batchSize)
		numBatches := tdg.NumBatches()
		isLoggingEpoch := ((epoch%params.LogEvery == 0) || (epoch == params.Epochs) || (epoch == 1))
		logEveryBatch := numBatches / 100
		if logEveryBatch == 0 {
			logEveryBatch = 1
		}
		loss := 0.0
		currentBatches := 0.0
		bi := 0
		for {
			xBatch, yBatch, err := tdg.NextBatch()
			if err != nil {
				return err
			}
			if xBatch == nil || yBatch == nil {
				break
			}
			batchLoss, err := m.FitBatch(xBatch, yBatch, solver)
			if err != nil {
				return err
			}
			loss += batchLoss
			currentBatches++
			if params.Verbose && isLoggingEpoch && bi%logEveryBatch == 0 {
				bar := strings.Repeat("=", int(currentBatches/float64(numBatches)*39))
				bar += ">"
				fmt.Printf("\rEpoch %d/%d - Loss: %f |%-40v|", epoch, params.Epochs, loss/currentBatches, bar)
			}
			bi++
		}
		if params.Verbose && isLoggingEpoch {
			lineEnd := "\n"
			if params.ClearLine {
				lineEnd = "\r"
			}
			fmt.Printf("\rEpoch %d/%d - Loss: %f |Done| %40v%v", epoch, params.Epochs, loss/currentBatches, "", lineEnd)
		}
		for _, cb := range params.EpochEndCallbakcs {
			if err := cb(epoch, loss/currentBatches); err != nil {
				return err
			}
		}
	}
	if params.Verbose {
		fmt.Println()
	}
	return nil
}

// MustFitGenerator calls FitGenerator, but panics if there is an error.
func (m *Model) MustFitGenerator(tdg TrainingDataGenerator, solver G.Solver, opts ...FitOpt) {
	err := m.FitGenerator(tdg, solver, opts...)
	if err != nil {
		panic(err)
	}
}

// Predict returns the models outputs for the given inputs. It cuts the inputs into batches so the inputs can be of any length.
func (m *Model) Predict(xs map[string]T.Tensor) (map[string]T.Tensor, error) {
	xBatchess, numPads, err := batchMultipleTensors(xs, m.getCurrentBatchSize(), true)
	if err != nil {
		return nil, err
	}
	yBatchess := make([]map[string]T.Tensor, len(xBatchess))
	for bi := range xBatchess {
		yBatches, err := m.PredictBatch(xBatchess[bi])
		if err != nil {
			return nil, err
		}
		// Remove padding
		if bi == len(xBatchess)-1 {
			for name := range yBatches {
				yBatches[name], err = sliceBatch(yBatches[name], T.S(0, yBatches[name].Shape()[0]-numPads))
				if err != nil {
					return nil, err
				}
			}
		}
		yBatchess[bi] = yBatches
	}
	// Concatenate the batches back together
	ys := make(map[string]T.Tensor, 0)
	for name := range yBatchess[0] {
		batchesForOutput := make([]T.Tensor, 0)
		for batch := range yBatchess {
			batchesForOutput = append(batchesForOutput, yBatchess[batch][name])
		}
		y, err := T.Concat(0, batchesForOutput[0], batchesForOutput[1:]...)
		if err != nil {
			return nil, err
		}
		ys[name] = y

	}
	return ys, nil
}

// MustPredict calls Predict, but panics if there is an error.
func (m *Model) MustPredict(xs map[string]T.Tensor) map[string]T.Tensor {
	ys, err := m.Predict(xs)
	if err != nil {
		panic(err)
	}
	return ys
}

func (m *Model) getCurrentBatchSize() int {
	for _, n := range m.InputNodes {
		return n.Shape()[0]
	}
	panic("this shouldn't be possible to reach, do you have no input nodes for some reason?")
}

// Creates a list of batches from the data. The data is a slice of tensors, representing multiple inputs.
// If zeroPadding is true, the last batch will be padded with zeros if it is smaller than the batch size.
// If zeroPadding is false, the last batch will be discarded if it is smaller than the batch size.
// Takes input [input_num]Tensor and returns [batch][input_num]Tensor
func batchMultipleTensors(inputs map[string]T.Tensor, batchSize int, zeroPad bool) ([]map[string]T.Tensor, int, error) {
	numRows := -1
	for _, input := range inputs {
		if numRows == -1 {
			numRows = input.Shape()[0]
		} else if numRows != input.Shape()[0] {
			return nil, 0, fmt.Errorf("all inputs must have the same number of rows")
		}
	}
	remainder := numRows % batchSize
	numNeededBatch := batchSize - remainder
	if remainder == 0 {
		numNeededBatch = 0
	}
	// We need to copy so we dont modify the inputs array. This does not do tensor copying, just the slice
	paddedInputs := make(map[string]T.Tensor, len(inputs))
	copyMap(paddedInputs, inputs)
	// If we have a number of inputs that does not perfectly fit, either pad or cut off the remainder
	if remainder != 0 {
		if zeroPad {
			// Pad the inputs so the remainder is part of a batch
			for name := range paddedInputs {
				paddingShape := append([]int{numNeededBatch}, paddedInputs[name].Shape()[1:]...)
				padding := T.New(T.WithShape(paddingShape...), T.Of(paddedInputs[name].Dtype()))
				var err error
				paddedInputs[name], err = T.Concat(0, paddedInputs[name], padding)
				if err != nil {
					return nil, 0, err
				}
			}
		} else {
			// Cut off the remainder
			for inputI := range paddedInputs {
				var err error
				paddedInputs[inputI], err = sliceBatch(paddedInputs[inputI], T.S(0, numRows-remainder))
				if err != nil {
					return nil, 0, err
				}
			}
		}
	}
	var batchedInputs []map[string]T.Tensor
	numPaddedRows := -1
	for _, input := range paddedInputs {
		numPaddedRows = input.Shape()[0]
		break
	}
	numBatches := numPaddedRows / batchSize
	for batchI := 0; batchI < numBatches; batchI += 1 {
		batch := map[string]T.Tensor{}
		for inputName, input := range paddedInputs {
			batchStart := batchI * batchSize
			slice, err := sliceBatch(input, T.S(batchStart, batchStart+batchSize))
			if err != nil {
				panic(err) // TODO - handle this error
			}
			batch[inputName] = slice
		}
		batchedInputs = append(batchedInputs, batch)
	}
	return batchedInputs, numNeededBatch, nil
}

// This performs a slice on the first dimension but guarantees that the output will have same ndims as input
func sliceBatch(t T.Tensor, slice T.Slice) (T.Tensor, error) {
	origShape := t.Shape()
	st, err := t.Slice(slice)
	if err != nil {
		return nil, err
	}
	if len(st.Shape()) != len(origShape) {
		newShape := origShape
		newShape[0] = 1
		err = st.Reshape(newShape...)
		if err != nil {
			return nil, err
		}
	}
	return st, nil
}

// Summary returns a string summarising the model.
func (m *Model) Summary() string {
	s := ""
	s += "================== Inputs ===================\n"
	for name, node := range m.InputNodes {
		s += fmt.Sprintf("Input       %-20v          Shape: %-20v\n", name, fmt.Sprint(node.Shape()))
	}
	s += "================== Outputs ==================\n"
	for name, node := range m.OutputNodes {
		s += fmt.Sprintf("Output      %-20v          Shape: %-20v\n", name, fmt.Sprint(node.Shape()))
	}
	s += "================= Loss Reqs =================\n"
	for name, node := range m.LossRequiredNodes {
		s += fmt.Sprintf("Loss Req    %-20v          Shape: %-20v\n", name, fmt.Sprint(node.Shape()))
	}
	totalParams := 0
	s += "============= Registered Layers =============\n"
	for li := range m.Layers {
		reqs := make([]string, 0)
		for _, r := range m.Layers[li].INodes() {
			reqs = append(reqs, r.Name())
		}
		numParams := 0
		for _, p := range m.Layers[li].Parameters() {
			numParams += p.DataSize()
		}
		totalParams += numParams
		s += fmt.Sprintf("Layer %-3v %9v::%-21vShape: %-20v From: %-20v Num Params %v\n",
			li, m.Layers[li].Name(), m.Layers[li].Type(),
			fmt.Sprint(m.Layers[li].Node().Shape()),
			reqs, numParams)
	}
	s += "=================== Stats ===================\n"
	s += fmt.Sprintf("Total number of parameters: %v\n", totalParams)
	return s
}
