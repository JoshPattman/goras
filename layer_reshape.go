package goras

import (
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// ReshapeLayer is a reshape layer.
//   - Input Shape: any shape
//   - Output Shape: the specified shape [as long as both shapes have the same volume]
type ReshapeLayer struct {
	LayerBase
	ToShape T.Shape
}

// Reshape creates a new ReshapeLayer on the Model with the given target shape.
func Reshape(model *Model, name string, newShape T.Shape) *ReshapeLayer {
	l := &ReshapeLayer{
		LayerBase: LayerBase{model.Graph, name, "reshape", false, model.DType, nil, nil},
		ToShape:   newShape,
	}
	model.AddLayer(l)
	return l
}

// Attach attaches the ReshapeLayer to the given node.
func (l *ReshapeLayer) Attach(n *G.Node) (*G.Node, error) {
	if err := validateShape(n.Shape(), valMatchingVolume(l.ToShape)); err != nil {
		return nil, err
	}
	on, err := G.Reshape(n, l.ToShape)
	l.OutputNode = on
	if on != nil {
		G.WithName(l.Name() + ".reshape")(on)
	}
	l.InputNodes = []*G.Node{n}
	return on, err
}

// MustAttach attaches the ReshapeLayer to the given node. It panics on error.
func (l *ReshapeLayer) MustAttach(n *G.Node) *G.Node { return mustAttach(l, n) }

// Parameters returns a map of the parameters of the layer.
func (l *ReshapeLayer) Parameters() map[string]*G.Node {
	return make(map[string]*G.Node)
}
