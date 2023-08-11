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
	Node() *G.Node                  // This returns the node used as the main output for this layer
	INodes() []*G.Node              // This returns all nodes used as inputs to this layer
}

// LayerBase is a struct that all layers should embed.
// It provides some useful shared fields and methods.
type LayerBase struct {
	Graph       *G.ExprGraph
	LayerName   string
	LayerType   string
	IsTrainable bool
	DType       T.Dtype
	OutputNode  *G.Node
	InputNodes  []*G.Node
}

// Name returns the name of the layer (e.g. "model_1").
func (l *LayerBase) Name() string {
	return l.LayerName
}

// Type returns the type of the layer (e.g. "dense").
func (l *LayerBase) Type() string {
	return l.LayerType
}

// Trainable returns whether the layer is trainable at the moment.
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

// Stuff for reducing repetitive code
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
