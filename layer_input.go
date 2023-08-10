package goras

import G "gorgonia.org/gorgonia"

// InputLayer is a layer that takes an input of a specific shape.
//   - Input/Output Shape: (batch_size, ...other_dims) [the specified shape]
type InputLayer struct {
	LayerBase
	Node *G.Node
}

// Input creates a new input layer on the specified model.
// To access the resulting *Node, use the .Node field.
func Input(m *Model, name string, shape ...int) *InputLayer {
	if err := validateShape(shape, valAtLeastNDims(1)); err != nil {
		panic(err)
	}
	i := &InputLayer{LayerBase{m.Graph, name, false, m.DType}, G.NewTensor(m.Graph, m.DType, len(shape), G.WithShape(shape...))}
	m.AddLayer(i)
	return i
}

// Parameters returns a map of the parameters of the layer.
func (l *InputLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }

func (l *InputLayer) Type() string { return "input" }
