package goras

import G "gorgonia.org/gorgonia"

// InputLayer is a layer that takes an input of a specific shape.
type InputLayer struct {
	LayerBase
	Node *G.Node
}

// Attach attaches the layer to a previous node.
// It then returns the node that the layer outputs.
// The shape shoudl include the batch size.
// For example, if the input is a 2D image, the shape should be [batchSize, width, height, channels].
func Input(m *Model, name string, shape ...int) *InputLayer {
	if err := validateShape(shape, valAtLeastNDims(1)); err != nil {
		panic(err)
	}
	i := &InputLayer{LayerBase{m.Graph, name, false}, G.NewTensor(m.Graph, G.Float64, len(shape), G.WithShape(shape...))}
	m.AddLayer(i)
	return i
}

// Parameters returns a map of the parameters of the layer.
func (i *InputLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }
