package goras

import (
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

type ReshapeLayer struct {
	LayerBase
	ToShape T.Shape
}

func Reshape(model *Model, name string, newShape T.Shape) *ReshapeLayer {
	return &ReshapeLayer{
		LayerBase: LayerBase{model.Graph, name, false},
		ToShape:   newShape,
	}
}

func (l *ReshapeLayer) Attach(n *G.Node) (*G.Node, error) {
	return G.Reshape(n, l.ToShape)
}

func (l *ReshapeLayer) MustAttach(n *G.Node) *G.Node {
	n, err := l.Attach(n)
	if err != nil {
		panic(err)
	}
	return n
}
