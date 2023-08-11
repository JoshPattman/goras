package goras

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// ActivationLayer is a layer that applies an activation function to its input.
//   - Input/Output Shape: any shape
type ActivationLayer struct {
	LayerBase
	Activation    string
	LeakyReluGrad float64
}

// Activation creates a new ActivationLayer on the Model with the given activation function.
// The activation function can be one of ["sigmoid", "relu", "tanh", "binary", "softmax", "leakyrelu"].
func Activation(m *Model, name string, activation string) *ActivationLayer {
	a := &ActivationLayer{LayerBase{m.Graph, name, "activation(" + activation + ")", false, m.DType, nil}, activation, 0.01}
	m.AddLayer(a)
	return a
}

func Sigmoid(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "sigmoid")
}

func Relu(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "relu")
}

func Tanh(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "tanh")
}

func Binary(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "binary")
}

func Softmax(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "softmax")
}

func LeakyRelu(m *Model, name string, grad ...float64) *ActivationLayer {
	a := Activation(m, name, "leakyrelu")
	if len(grad) > 0 {
		a.LeakyReluGrad = grad[0]
	}
	return a
}

// Attach attaches this layer to a previous node.
func (l *ActivationLayer) Attach(n *G.Node) (*G.Node, error) {
	var on *G.Node
	var err error
	switch l.Activation {
	case "sigmoid":
		on, err = G.Sigmoid(n)
	case "relu":
		on, err = G.Rectify(n)
	case "tanh":
		on, err = G.Tanh(n)
	case "binary":
		on, err = G.Gt(n, G.NewConstant(0.0, G.WithType(l.DType)), true)
	case "softmax":
		on, err = G.SoftMax(n, 1)
	case "leakyrelu":
		on, err = G.LeakyRelu(n, 0.01)
	default:
		return nil, fmt.Errorf("invalid activation '%s'", l.Activation)
	}
	l.OutputNode = on
	if on != nil {
		G.WithName(l.Name() + ".activation")(on)
	}
	return on, err
}

func (l *ActivationLayer) MustAttach(n *G.Node) *G.Node { return mustAttach(l, n) }

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
