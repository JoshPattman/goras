package goras

import (
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// DenseLayer is a layer that performs a dense (fully connected) operation.
// It does not perform any activation or dropout.
//   - Input Shape: (batch_size, num_inputs)
//   - Output Shape: (batch_size, num_nodes)
type DenseLayer struct {
	LayerBase
	Weights *G.Node
	Nodes   int
}

// Dense creates a new dense layer on the specified model.
func Dense(m *Model, name string, nodes int) *DenseLayer {
	d := &DenseLayer{LayerBase{m.Graph, name, "dense", true, nil, nil}, nil, nodes}
	m.AddLayer(d)
	return d
}

// Attach attaches the layer to a previous node.
func (l *DenseLayer) Attach(n *G.Node) (*G.Node, error) {
	if err := validateShape(n.Shape(), valNDims(2)); err != nil {
		return nil, err
	}
	numInputs := n.Shape()[1]
	batchSize := n.Shape()[0]
	l.Weights = G.NewMatrix(l.Graph, n.Dtype(), G.WithShape(numInputs+1, l.Nodes), G.WithInit(G.GlorotN(1.0)), G.WithName(l.Name()+".weights"))
	bias := G.NewConstant(T.Ones(n.Dtype(), batchSize, 1), G.WithName(l.Name()+".bias"))
	// Build the graph
	withBias, err := G.Concat(1, n, bias)
	if err != nil {
		return nil, err
	}
	multiplied, err := G.Mul(withBias, l.Weights)
	if err != nil {
		return nil, err
	}
	l.OutputNode = multiplied
	if l.OutputNode != nil {
		G.WithName(l.Name() + ".matmul")(l.OutputNode)
	}
	l.InputNodes = []*G.Node{n}
	return multiplied, nil
}

// MustAttach attaches the layer to a previous node, panicking on error.
func (l *DenseLayer) MustAttach(n *G.Node) *G.Node { return mustAttach(l, n) }

// Parameters returns a map of the parameters of the layer.
func (l *DenseLayer) Parameters() map[string]*G.Node {
	return map[string]*G.Node{"weights": l.Weights}
}
