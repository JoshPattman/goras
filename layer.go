package goras

import (
	G "gorgonia.org/gorgonia"
)

// Layer is an interface that all layers must implement to be able to be added to a model.
type Layer interface {
	Parameters() map[string]*G.Node
	Name() string
}

// LayerBase is a struct that all layers should embed.
// It provides a Graph and a name, and implements the Name() method of the Layer interface.
type LayerBase struct {
	Graph     *G.ExprGraph
	LayerName string
}

// Name returns the name of the layer.
func (l *LayerBase) Name() string {
	return l.LayerName
}
