package goras

import (
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// DenseLayer is a layer that performs a dense (fully connected) operation.
// It does not perform any activation or dropout.
type DenseLayer struct {
	LayerBase
	Weights *G.Node
	Nodes   int
}

// Dense creates a new dense layer on the specified model.
// The layer will have the specified number of nodes, and therefor the output of the layer will have that many nodes.
func Dense(m *Model, name string, nodes int) *DenseLayer {
	d := &DenseLayer{LayerBase{m.Graph, name, true}, nil, nodes}
	m.AddLayer(d)
	return d
}

// Attach attaches the layer to a previous node.
// It then returns the node that the layer outputs.
func (d *DenseLayer) Attach(n *G.Node) (*G.Node, error) {
	if d.Weights != nil {
		panic("Already attached")
	}
	numInputs := n.Shape()[1]
	batchSize := n.Shape()[0]
	d.Weights = G.NewMatrix(d.Graph, G.Float64, G.WithShape(numInputs+1, d.Nodes), G.WithInit(G.GlorotN(1.0)))
	bias := G.NewConstant(T.Ones(T.Float64, batchSize, 1))
	// Build the graph
	withBias, err := G.Concat(1, n, bias)
	if err != nil {
		return nil, err
	}
	multiplied, err := G.Mul(withBias, d.Weights)
	if err != nil {
		return nil, err
	}
	return multiplied, nil
}

// Attach attaches the layer to a previous node.
// It then returns the node that the layer outputs.
// Panics if there is an error.
func (d *DenseLayer) MustAttach(n *G.Node) *G.Node {
	n, err := d.Attach(n)
	if err != nil {
		panic(err)
	}
	return n
}

// Parameters returns a map of the parameters of the layer.
func (d *DenseLayer) Parameters() map[string]*G.Node {
	return map[string]*G.Node{"weights": d.Weights}
}
