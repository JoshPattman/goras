package goras

import (
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// InputLayer is a layer that takes an input of a specific shape.
//   - Input/Output Shape: (batch_size, ...other_dims) [the specified shape]
type InputLayer struct {
	LayerBase
}

// Input creates a new input layer on the specified model.
// To access the resulting *Node, use the .Node() function.
func Input(m *Model, name string, dtype T.Dtype, shape ...int) *InputLayer {
	if err := validateShape(shape, valAtLeastNDims(1)); err != nil {
		panic(err)
	}
	t := G.NewTensor(m.Graph, dtype, len(shape), G.WithShape(shape...), G.WithName(name+".input"))
	i := &InputLayer{LayerBase{m.Graph, name, "input", false, t, nil}}
	m.AddLayer(i)
	i.InputNodes = []*G.Node{}
	return i
}

// Parameters returns a map of the parameters of the layer.
func (l *InputLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }
