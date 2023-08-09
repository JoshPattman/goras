package goras

import (
	G "gorgonia.org/gorgonia"
)

type DropoutLayer struct {
	LayerBase
	DropoutProbability float64
}

func Dropout(m *Model, name string, dropoutProbability float64) *DropoutLayer {
	d := &DropoutLayer{LayerBase{m.Graph, name, false}, dropoutProbability}
	m.AddLayer(d)
	return d
}
func (d *DropoutLayer) Attach(n *G.Node) (*G.Node, error) {
	return G.Dropout(n, d.DropoutProbability)
}
func (d *DropoutLayer) MustAttach(n *G.Node) *G.Node {
	n, err := d.Attach(n)
	if err != nil {
		panic(err)
	}
	return n
}

func (d *DropoutLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }
