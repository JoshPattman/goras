package goras

import (
	G "gorgonia.org/gorgonia"
)

// DropoutLayer is a dropout layer.
//   - Input/Output Shape: any shape
type DropoutLayer struct {
	LayerBase
	DropoutProbability float64
}

// Dropout creates a new DropoutLayer on the Model with the given dropout probability.
func Dropout(m *Model, name string, dropoutProbability float64) *DropoutLayer {
	d := &DropoutLayer{LayerBase{m.Graph, name, "dropout", false, m.DType, nil, nil}, dropoutProbability}
	m.AddLayer(d)
	return d
}

// Attach attaches the DropoutLayer to the given node.
func (l *DropoutLayer) Attach(n *G.Node) (*G.Node, error) {
	on, err := G.Dropout(n, l.DropoutProbability)
	l.OutputNode = on
	if on != nil {
		G.WithName(l.Name() + ".dropout")(on)
	}
	l.InputNodes = []*G.Node{n}
	return on, err
}

// MustAttach attaches the DropoutLayer to the given node. It panics on error.
func (l *DropoutLayer) MustAttach(n *G.Node) *G.Node { return mustAttach(l, n) }

// Parameters returns a map of the parameters of the layer.
func (d *DropoutLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }
