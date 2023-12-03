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
	a := &ActivationLayer{LayerBase{m.Graph, name, "activation(" + activation + ")", false, m.DType, nil, nil}, activation, 0.01}
	m.AddLayer(a)
	return a
}

// Sigmoid creates a new ActivationLayer on the Model with the sigmoid activation function.
func Sigmoid(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "sigmoid")
}

// Relu creates a new ActivationLayer on the Model with the relu activation function.
func Relu(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "relu")
}

// Tanh creates a new ActivationLayer on the Model with the tanh activation function.
func Tanh(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "tanh")
}

// Binary creates a new ActivationLayer on the Model with the binary activation function.
func Binary(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "binary")
}

// Softmax creates a new ActivationLayer on the Model with the softmax activation function.
func Softmax(m *Model, name string) *ActivationLayer {
	return Activation(m, name, "softmax")
}

// LeakyRelu creates a new ActivationLayer on the Model with the leaky relu activation function.
// You can optionally specify the negative gradient (LeakyRely(model, name, grad)).
// If you don't, it will default to 0.01.
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
		on, err = customSoftMax(n) //G.SoftMax(n, 1) // TODO: my custom softmax seems to be working but gorgonias dosn't. Invistigate more and maybe create an issue.
	case "leakyrelu":
		on, err = G.LeakyRelu(n, 0.01)
	default:
		return nil, fmt.Errorf("invalid activation '%s'", l.Activation)
	}
	l.OutputNode = on
	if on != nil {
		G.WithName(l.Name() + ".activation")(on)
	}
	l.InputNodes = []*G.Node{n}
	return on, err
}

// MustAttach attaches this layer to a previous node. It panics on error.
func (l *ActivationLayer) MustAttach(n *G.Node) *G.Node { return mustAttach(l, n) }

// Parameters returns a map of the parameters of the layer.
func (l *ActivationLayer) Parameters() map[string]*G.Node { return make(map[string]*G.Node) }

// This function is designed to be a drop in replacement for G.SoftMax.
// This is to try and find the dreaded softmax panic.
// It will also only do stuff on axis 1
// Also, this is probably slower than the built in softmax function as it uses mutiple nodes.
func customSoftMax(x *G.Node) (*G.Node, error) {
	var err error
	exponentiatedClasses, err := G.Exp(x)
	if err != nil {
		return nil, err
	}
	summedExponentiatedClasses, err := G.Sum(exponentiatedClasses, 1)
	if err != nil {
		return nil, err
	}
	return G.BroadcastHadamardDiv(exponentiatedClasses, summedExponentiatedClasses, []byte{}, []byte{1})
}
