package goras

import (
	G "gorgonia.org/gorgonia"
)

// Layer is an interface that all layers must implement to be able to be added to a model.
type Layer interface {
	Parameters() map[string]*G.Node // This returns a map of the parameters. E.g. {"weights":[...], "biases":[...]}
	Name() string                   // This returns a name unique to this layer in the model
	Trainable() bool                // This specifies whether the layer is updated during Fit()
	Type() string                   // This is used for Summary()
}

// LayerBase is a struct that all layers should embed.
// It provides a Graph and a name, and implements the Name() and Trainable methods of the Layer interface.
type LayerBase struct {
	Graph       *G.ExprGraph
	LayerName   string
	IsTrainable bool
}

// Name returns the name of the layer.
func (l *LayerBase) Name() string {
	return l.LayerName
}

func (l *LayerBase) Trainable() bool {
	return l.IsTrainable
}
