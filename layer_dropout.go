package goras

import (
	G "gorgonia.org/gorgonia"
)

type DropoutLayer struct {
	LayerBase
	DropoutProbability float64
}

func Dropout(m *Model, name string, dropoutProbability float64) *DropoutLayer {
	d := &DropoutLayer{LayerBase{m.Graph, name}, dropoutProbability}
	m.AddLayer(d)
	return d
}
func (d *DropoutLayer) Attach(n *G.Node) *G.Node {
	return G.Must(G.Dropout(n, d.DropoutProbability))
}

func (d *DropoutLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }
