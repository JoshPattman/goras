package goras

import G "gorgonia.org/gorgonia"

// ActivationLayer is a layer that applies an activation function to its input.
type ActivationLayer struct {
	LayerBase
	Activation string
}

// Activation creates a new ActivationLayer on the Model with the given activation function.
// The activation function can be either "sigmoid" or "relu".
func Activation(m *Model, name string, activation string) *ActivationLayer {
	a := &ActivationLayer{LayerBase{m.Graph, name}, activation}
	m.AddLayer(a)
	return a
}

// Attach attaches the layer to a previous node.
// It then returns the node that the layer outputs.
func (a *ActivationLayer) Attach(n *G.Node) *G.Node {
	switch a.Activation {
	case "sigmoid":
		return G.Must(G.Sigmoid(n))
	case "relu":
		return G.Must(G.Rectify(n))
	default:
		panic("Invalid activation")
	}
}

// Parameters returns a map of the parameters of the layer.
func (a *ActivationLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }
