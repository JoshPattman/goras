package goras

import (
	G "gorgonia.org/gorgonia"
)

type DropoutLayer struct {
	LayerBase
	DropoutProbability float64
}

func Dropout(m *Model, name string, dropoutProbability float64) *DropoutLayer {
	d := &DropoutLayer{LayerBase{m.Graph, name, false, m.DType}, dropoutProbability}
	m.AddLayer(d)
	return d
}
func (l *DropoutLayer) Attach(n *G.Node) (*G.Node, error) {
	return G.Dropout(n, l.DropoutProbability)
}
func (l *DropoutLayer) MustAttach(n *G.Node) *G.Node {
	n, err := l.Attach(n)
	if err != nil {
		panic(err)
	}
	return n
}

func (d *DropoutLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }

func (d *DropoutLayer) Type() string { return "dropout" }
