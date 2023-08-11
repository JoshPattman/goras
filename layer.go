package goras

import (
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// Layer is an interface that all layers must implement to be able to be added to a model.
type Layer interface {
	Parameters() map[string]*G.Node // This returns a map of the parameters. E.g. {"weights":[...], "biases":[...]}
	Name() string                   // This returns a name unique to this layer in the model
	Trainable() bool                // This specifies whether the layer is updated during Fit()
	Type() string                   // This is used for Summary()
	Node() *G.Node                  // Returns the output node
	INodes() []*G.Node              // Returns the input nodes
}

// LayerBase is a struct that all layers should embed.
// It provides a Graph and a name, and implements the Name() and Trainable methods of the Layer interface.
type LayerBase struct {
	Graph       *G.ExprGraph
	LayerName   string
	LayerType   string
	IsTrainable bool
	DType       T.Dtype
	OutputNode  *G.Node
	InputNodes  []*G.Node
}

// Name returns the name of the layer.
func (l *LayerBase) Name() string {
	return l.LayerName
}

func (l *LayerBase) Type() string {
	return l.LayerType
}

func (l *LayerBase) Trainable() bool {
	return l.IsTrainable
}

// Node returns the final node in this layer (the output node)
func (l *LayerBase) Node() *G.Node {
	return l.OutputNode
}

// INodes returns the input nodes of this layer.
func (l *LayerBase) INodes() []*G.Node {
	return l.InputNodes
}

type attacher interface {
	Attach(*G.Node) (*G.Node, error)
}

func mustAttach(l attacher, x *G.Node) *G.Node {
	n, err := l.Attach(x)
	if err != nil {
		panic(err)
	}
	return n
}
