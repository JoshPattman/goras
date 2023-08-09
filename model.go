package goras

import (
	"encoding/gob"
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
func (m *Model) Build(inputNode, outputNode *G.Node, loss func(*G.Node, *G.Node) *G.Node) {
	// Store input and output nodes
	m.InputNode = inputNode
	m.OutputNode = outputNode
	// Read the output to a value
	G.Read(m.OutputNode, &m.OutputValue)
	// Define loss function
	m.TargetOutputNode = G.NewMatrix(m.Graph, G.Float64, G.WithShape(m.OutputNode.Shape()...))
	lossNode := loss(m.OutputNode, m.TargetOutputNode)
	G.Read(lossNode, &m.LossValue)
	G.Grad(lossNode, m.Trainables()...)

	// Create machine
	m.Machine = G.NewTapeMachine(m.Graph)
}

// Trainables returns a list of all the trainable nodes in the model.
func (m *Model) Trainables() G.Nodes {
	var ret G.Nodes
	for _, l := range m.Layers {
		for _, t := range l.Parameters() {
			ret = append(ret, t)
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