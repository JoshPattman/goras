package goras

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// ActivationLayer is a layer that applies an activation function to its input.
//   - Input/Output Shape: any shape
type ActivationLayer struct {
	LayerBase
	Activation string
}

// Activation creates a new ActivationLayer on the Model with the given activation function.
// The activation function can be one of ["sigmoid", "relu", "tanh", "binary", "softmax", "leakyrelu"].
func Activation(m *Model, name string, activation string) *ActivationLayer {
	a := &ActivationLayer{LayerBase{m.Graph, name, false, m.DType}, activation}
	m.AddLayer(a)
	return a
}

// Attach attaches this layer to a previous node.
func (l *ActivationLayer) Attach(n *G.Node) (*G.Node, error) {
	switch l.Activation {
	case "sigmoid":
		return G.Sigmoid(n)
	case "relu":
		return G.Rectify(n)
	case "tanh":
		return G.Tanh(n)
	case "binary":
		return G.Gt(n, G.NewConstant(0.0, G.WithType(l.DType)), true)
	case "softmax":
		return G.SoftMax(n, 1)
	case "leakyrelu":
		return G.LeakyRelu(n, 0.01)
	default:
		return nil, fmt.Errorf("invalid activation '%s'", l.Activation)
	}
}

// MustAttach attaches this layer to a previous node and panics on error.
func (l *ActivationLayer) MustAttach(n *G.Node) *G.Node {
	n, err := l.Attach(n)
	if err != nil {
		panic(err)
	}
	return n
}

// Parameters returns a map of the parameters of the layer.
func (l *ActivationLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }

func IsValidActivation(ac string) bool {
	switch ac {
	case "sigmoid", "relu", "tanh", "binary", "softmax", "leakyrelu":
		return true
	default:
		return false
	}
}

// Type returns the type of the layer as a string.
func (l *ActivationLayer) Type() string { return "activation" }
