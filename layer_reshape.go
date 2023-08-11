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

func Reshape(model *Model, name string, newShape T.Shape) *ReshapeLayer {
	l := &ReshapeLayer{
		LayerBase: LayerBase{model.Graph, name, "reshape", false, model.DType, nil},
		ToShape:   newShape,
	}
	model.AddLayer(l)
	return l
}

func (l *ReshapeLayer) Attach(n *G.Node) (*G.Node, error) {
	if err := validateShape(n.Shape(), valMatchingVolume(l.ToShape)); err != nil {
		return nil, err
	}
	on, err := G.Reshape(n, l.ToShape)
	l.OutputNode = on
	return on, err
}

func (l *ReshapeLayer) MustAttach(n *G.Node) *G.Node {
	n, err := l.Attach(n)
	if err != nil {
		panic(err)
	}
	return n
}

func (l *ReshapeLayer) Parameters() map[string]*G.Node {
	return make(map[string]*G.Node)
}
