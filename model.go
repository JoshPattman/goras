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
	InputNode        *G.Node
	OutputNode       *G.Node
	OutputValue      G.Value
	LossValue        G.Value
	TargetOutputNode *G.Node
}

// NewModel creates a new model with no layers
func NewModel() *Model {
	return &Model{Graph: G.NewGraph(), Layers: []Layer{}}
}

// AddLayer adds a layer to the model. You usually don't need to call this directly, as the layer constructors do it for you.
func (m *Model) AddLayer(l Layer) {
	m.Layers = append(m.Layers, l)
}

// Build builds the model, using a specified input and output node.
// It adds the loss function to the graph, and creates the machine.
// This should only be called once per model.
func (m *Model) Build(inputNode, outputNode *G.Node, loss func(*G.Node, *G.Node) (*G.Node, error)) error {
	// Store input and output nodes
	m.InputNode = inputNode
	m.OutputNode = outputNode
	// Read the output to a value
	G.Read(m.OutputNode, &m.OutputValue)
	// Define loss function
	m.TargetOutputNode = G.NewMatrix(m.Graph, G.Float64, G.WithShape(m.OutputNode.Shape()...))
	lossNode, err := loss(m.OutputNode, m.TargetOutputNode)
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

func (m *Model) MustBuild(inputNode, outputNode *G.Node, loss func(*G.Node, *G.Node) (*G.Node, error)) {
	err := m.Build(inputNode, outputNode, loss)
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
func (m *Model) SetParams(params map[string]*T.Dense) {
	for _, l := range m.Layers {
		for k, v := range l.Parameters() {
			if p, ok := params[l.Name()+":"+k]; ok {
				G.Let(v, p)
			}
		}
	}
}

// WriteParams writes the parameters in gob format to an io.Writer.
// The params are retrieved with Model.GetParams.
func (m *Model) WriteParams(w io.Writer) {
	params := m.GetParams()
	enc := gob.NewEncoder(w)
	if err := enc.Encode(params); err != nil {
		panic(err)
	}
}

// ReadParams reads the parameters in gob format from an io.Reader.
// The params are retrieved with Model.GetParams.
func (m *Model) ReadParams(r io.Reader) {
	var params map[string]*T.Dense
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&params); err != nil {
		panic(err)
	}
	m.SetParams(params)
}

// BindParamsFrom binds the parameters in the model to the parameters in another model, meaning layers with the same name will share the same tensors.
// This is a bit of a hack to allow two models to train the same weights.
// This can be called multiple times, where later binds may override earlier ones.
// For example, if you are making an autoencoder, you would have one main model for training, and an encoder model and decoder model which are bound to that.
// That then allows you to run partial bits of the network.
func (m *Model) BindParamsFrom(m1 *Model) {
	paramsSrc := m1.GetParams()
	for _, l := range m.Layers {
		for k, v := range l.Parameters() {
			if p, ok := paramsSrc[l.Name()+":"+k]; ok {
				G.Let(v, p)
			}
		}
	}
}

// PredictBatch runs the model on a batch of input data. The batch size must match the input node shape.
func (m *Model) PredictBatch(input *T.Dense) *T.Dense {
	m.Machine.Reset()
	G.Let(m.InputNode, input)
	G.Let(m.TargetOutputNode, T.New(T.WithShape(m.TargetOutputNode.Shape()...), T.WithBacking(make([]float64, m.TargetOutputNode.Shape().TotalSize()))))
	if err := m.Machine.RunAll(); err != nil {
		panic(err)
	}
	return T.New(T.WithShape(m.OutputValue.Shape()...), T.WithBacking(m.OutputValue.Data()))
}

// FitBatch runs the model on a batch of input data, and then trains the model on the target data.
// The solver used is passed in as an argument.
// IMPORTANT NOTE: Currently, when the data is batched, the last batch of data will be discarded if the x size does not evenly divide the batch size.
func (m *Model) FitBatch(input, target *T.Dense, solver G.Solver) float64 {
	m.Machine.Reset()
	G.Let(m.InputNode, input)
	G.Let(m.TargetOutputNode, target)
	if err := m.Machine.RunAll(); err != nil {
		panic(err)
	}
	solver.Step(G.NodesToValueGrads(m.Trainables()))
	return m.LossValue.Data().(float64)
}

// Creates a list of batches from the data. TODO - this copies the data, but i think it would be better to slice it.
func batchData(d *T.Dense, batchSize int) []*T.Dense {
	var ret []*T.Dense
	for i := 0; i < d.Shape()[0]; i += batchSize {
		if i+batchSize <= d.Shape()[0] {
			// This is a full batch
			slice, err := d.Slice(T.S(i, i+batchSize))
			if err != nil {
				panic(err)
			}
			// Now we need to create a dense copy of the slice.
			denseData := make([]float64, slice.Shape().TotalSize())
			copy(denseData, slice.Data().([]float64))
			denseSlice := T.New(T.WithShape(slice.Shape()...), T.WithBacking(denseData))
			ret = append(ret, denseSlice)
		}

	}
	return ret
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

func (m *Model) Fit(x, y *T.Dense, solver G.Solver, opts ...FitOpt) {
	params := &fitParams{
		Epochs:    1,
		LogEvery:  1,
		Verbose:   true,
		ClearLine: true,
	}
	for _, o := range opts {
		o(params)
	}
	batchSize := x.Shape()[0]
	xBatches, yBatches := batchData(x, batchSize), batchData(y, batchSize)
	for epoch := 0; epoch < params.Epochs; epoch++ {
		loss := 0.0
		for bi := range xBatches {
			loss += m.FitBatch(xBatches[bi], yBatches[bi], solver)
		}
		if params.Verbose && ((epoch%params.LogEvery == 0) || (epoch == params.Epochs-1)) {
			lineStart := "\n"
			if params.ClearLine {
				lineStart = "\r"
			}
			fmt.Printf("%sEpoch %d/%d - Loss: %f                    ", lineStart, epoch+1, params.Epochs, loss/float64(x.Shape()[0]))
		}
	}
}
