package goras

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// ActivationLayer is a layer that applies an activation function to its input.
type ActivationLayer struct {
	LayerBase
	Activation string
}

// Activation creates a new ActivationLayer on the Model with the given activation function.
// The activation function can be one of ["sigmoid", "relu", "tanh", "binary", "softmax"].
func Activation(m *Model, name string, activation string) *ActivationLayer {
	a := &ActivationLayer{LayerBase{m.Graph, name, false}, activation}
	m.AddLayer(a)
	return a
}

// Attach attaches the layer to a previous node.
// It then returns the node that the layer outputs.
func (a *ActivationLayer) Attach(n *G.Node) (*G.Node, error) {
	switch a.Activation {
	case "sigmoid":
		return G.Sigmoid(n)
	case "relu":
		return G.Rectify(n)
	case "tanh":
		return G.Tanh(n)
	case "binary":
		return G.Gt(n, G.NewConstant(0.0), true)
	case "softmax":
		return G.SoftMax(n, 1)
	default:
		return nil, fmt.Errorf("invalid activation '%s'", a.Activation)
	}
}

func (a *ActivationLayer) MustAttach(n *G.Node) *G.Node {
	n, err := a.Attach(n)
	if err != nil {
		panic(err)
	}
	return n
}

// Parameters returns a map of the parameters of the layer.
func (a *ActivationLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }

func IsValidActivation(ac string) bool {
	switch ac {
	case "sigmoid", "relu", "tanh", "binary", "softmax":
		return true
	default:
		return false
	}
}
