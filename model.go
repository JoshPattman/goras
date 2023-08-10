package goras

import (
	"encoding/gob"
	"fmt"
	"io"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// Model is a collection of layers which are all on the same graph,
// a machine which can be used to run the graph,
// and refrences to input, output and loss nodes.
type Model struct {
	Graph            *G.ExprGraph
	Layers           []Layer
	Machine          G.VM
	InputNodes       []*G.Node
	OutputNode       *G.Node
	OutputValue      G.Value
	LossValue        G.Value
	TargetOutputNode *G.Node
	DType            T.Dtype
}

// NewModel creates a new model with no layers
func NewModel(dataType T.Dtype) *Model {
	return &Model{Graph: G.NewGraph(), Layers: []Layer{}, DType: dataType}
}

// AddLayer adds a layer to the model. You usually don't need to call this directly, as the layer constructors do it for you.
func (m *Model) AddLayer(l Layer) {
	m.Layers = append(m.Layers, l)
}

type buildParams struct {
	inputNodes  []*G.Node
	outputNodes []*G.Node
	losses      []func(*G.Node, *G.Node) (*G.Node, error)
}
type BuildOpts func(*buildParams)

func WithInputs(inputNodes ...*G.Node) BuildOpts {
	return func(b *buildParams) { b.inputNodes = inputNodes }
}

func WithOutputs(outputNodes ...*G.Node) BuildOpts {
	return func(b *buildParams) { b.outputNodes = outputNodes }
}

func WithLosses(losses ...func(*G.Node, *G.Node) (*G.Node, error)) BuildOpts {
	return func(b *buildParams) { b.losses = losses }
}

// Build builds the model, using a specified input and output node.
// It adds the loss function to the graph, and creates the machine.
// This should only be called once per model.
func (m *Model) Build(opts ...BuildOpts) error {
	buildParams := &buildParams{}
	for _, opt := range opts {
		opt(buildParams)
	}
	if buildParams.inputNodes == nil || buildParams.outputNodes == nil || buildParams.losses == nil {
		return fmt.Errorf("inputNodes, outputNodes and losses must be specified")
	}
	if len(buildParams.outputNodes) != len(buildParams.losses) {
		return fmt.Errorf("outputNodes and losses must be the same length")
	}
	// For now, until i figure out multiple inputs and outputs, ensure there is examtly one input and output
	if len(buildParams.outputNodes) != 1 {
		return fmt.Errorf("only one output is supported at this time, this will change soon")
	}

	// Store input and output nodes
	m.InputNodes = buildParams.inputNodes
	m.OutputNode = buildParams.outputNodes[0]
	// Read the output to a value
	G.Read(m.OutputNode, &m.OutputValue)
	// Define loss function
	m.TargetOutputNode = G.NewMatrix(m.Graph, m.DType, G.WithShape(m.OutputNode.Shape()...))
	lossNode, err := buildParams.losses[0](m.OutputNode, m.TargetOutputNode)
	if err != nil {
		return err
	}
	G.Read(lossNode, &m.LossValue)
	_, err = G.Grad(lossNode, m.Trainables()...)
	if err != nil {
		return err
	}

	// Create machine
	m.Machine = G.NewTapeMachine(m.Graph)
	return nil
}

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

// WriteParams writes the parameters in gob format to an io.Writer.
// The params are retrieved with Model.GetParams.
func (m *Model) WriteParams(w io.Writer) error {
	params := m.GetParams()
	enc := gob.NewEncoder(w)
	return enc.Encode(params)
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

// BindParamsFrom binds the parameters in the model to the parameters in another model, meaning layers with the same name will share the same tensors.
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

// V is a helper function to create a slice of tensors. It should be used when providing a model with input and target data.
func V(ts ...T.Tensor) []T.Tensor { return ts }

// PredictBatch runs the model on a batch of input data. The batch size must match the input node shape.
func (m *Model) PredictBatch(inputs []T.Tensor) (*T.Dense, error) {
	if len(inputs) != len(m.InputNodes) {
		return nil, fmt.Errorf("number of inputs (%v) must be the same as number of input nodes (%v)", len(inputs), len(m.InputNodes))
	}
	if err := ensureCorrectBatchSize(inputs[0], m.getCurrentBatchSize()); err != nil {
		return nil, err
	}
	m.Machine.Reset()
	for i := range inputs {
		if err := G.Let(m.InputNodes[i], inputs[i]); err != nil {
			return nil, err
		}
	}
	if err := G.Let(m.TargetOutputNode, T.New(T.WithShape(m.TargetOutputNode.Shape()...), T.WithBacking(make([]float64, m.TargetOutputNode.Shape().TotalSize())))); err != nil {
		return nil, err
	}
	if err := m.Machine.RunAll(); err != nil {
		return nil, err
	}
	return T.New(T.WithShape(m.OutputValue.Shape()...), T.WithBacking(m.OutputValue.Data())), nil
}

// FitBatch runs the model on a batch of input data, and then trains the model on the target data.
// The solver used is passed in as an argument.
// IMPORTANT NOTE: Currently, when the data is batched, the last batch of data will be discarded if the x size does not evenly divide the batch size.
func (m *Model) FitBatch(inputs, targets []T.Tensor, solver G.Solver) (float64, error) {
	if len(inputs) != 1 || len(targets) != 1 {
		return 0, fmt.Errorf("number of inputs and targets must be 1 at this time")
	}
	target := targets[0]
	if err := ensureCorrectBatchSize(inputs[0], m.getCurrentBatchSize()); err != nil {
		return 0, err
	}
	m.Machine.Reset()
	for i := range inputs {
		if err := G.Let(m.InputNodes[i], inputs[i]); err != nil {
			return 0, err
		}
	}
	if err := G.Let(m.TargetOutputNode, target); err != nil {
		return 0, err
	}
	if err := m.Machine.RunAll(); err != nil {
		return 0, err
	}
	if err := solver.Step(G.NodesToValueGrads(m.Trainables())); err != nil {
		return 0, err
	}
	return m.LossValue.Data().(float64), nil
}

type FitOpt func(*fitParams)

type fitParams struct {
	Epochs    int
	LogEvery  int
	Verbose   bool
	ClearLine bool
}

// WithEpochs sets the number of epochs to train for.
func WithEpochs(epochs int) FitOpt { return func(p *fitParams) { p.Epochs = epochs } }

// WithLoggingEvery sets how often to log the loss.
func WithLoggingEvery(epochs int) FitOpt { return func(p *fitParams) { p.LogEvery = epochs } }

// WithVerbose sets whether to log the loss.
func WithVerbose(verbose bool) FitOpt { return func(p *fitParams) { p.Verbose = verbose } }

// WithClearLine sets whether to clear the line when logging the loss.
func WithClearLine(clear bool) FitOpt { return func(p *fitParams) { p.ClearLine = clear } }

func (m *Model) Fit(xs, ys []T.Tensor, solver G.Solver, opts ...FitOpt) error {
	params := &fitParams{
		Epochs:    1,
		LogEvery:  1,
		Verbose:   true,
		ClearLine: true,
	}
	for _, o := range opts {
		o(params)
	}
	batchSize := m.getCurrentBatchSize()
	// xBatchess and yBatchess are [batch][inputs]Tensor
	xBatchess, yBatchess := batchMultiData(xs, batchSize), batchMultiData(ys, batchSize)
	for epoch := 0; epoch < params.Epochs; epoch++ {
		loss := 0.0
		for bi := range xBatchess {
			batchLoss, err := m.FitBatch(xBatchess[bi], yBatchess[bi], solver)
			if err != nil {
				return err
			}
			loss += batchLoss
		}
		if params.Verbose && ((epoch%params.LogEvery == 0) || (epoch == params.Epochs-1)) {
			lineStart := "\n"
			if params.ClearLine {
				lineStart = "\r"
			}
			fmt.Printf("%sEpoch %d/%d - Loss: %f                    ", lineStart, epoch+1, params.Epochs, loss/float64(xs[0].Shape()[0]))
		}
	}
	if params.Verbose {
		fmt.Println()
	}
	return nil
}

func (m *Model) getCurrentBatchSize() int {
	return m.InputNodes[0].Shape()[0]
}

// Creates a list of batches from the data. The data is a slice of tensors, representing multiple inputs.
func batchMultiData(ds []T.Tensor, batchSize int) [][]T.Tensor {
	var ret [][]T.Tensor
	for i := 0; i < ds[0].Shape()[0]; i += batchSize {
		if i+batchSize <= ds[0].Shape()[0] { // TODO - This ignores the remainder - it should do somthing else
			// This is a full batch
			var batch []T.Tensor
			for _, d := range ds {
				slice, err := d.Slice(T.S(i, i+batchSize))
				if err != nil {
					panic(err) // TODO - handle this error
				}
				batch = append(batch, slice)
			}
			ret = append(ret, batch)
		}

	}
	return ret
}

func ensureCorrectBatchSize(batchData T.Tensor, batchSize int) error {
	if batchData.Shape()[0] != batchSize {
		return fmt.Errorf("incorrect batch size - expected %d, got %d", batchSize, batchData.Shape()[0])
	}
	return nil
}
