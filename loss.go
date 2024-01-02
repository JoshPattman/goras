package goras

import (
	G "gorgonia.org/gorgonia"
)

// LossFunc is a function that when called, returns:
//
// - a node (loss output scalar)
//
// - a map of nodes which the loss requires to be created (for instance, this is usually the target for the output layer)
//
// - an error
type LossFunc func() (lossOut *G.Node, lossInps map[string]*G.Node, err error)
