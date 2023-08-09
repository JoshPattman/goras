package goras

import G "gorgonia.org/gorgonia"

// MSE created the nodes to calculate mean squared error between a predicted and target node.
// It should be used when using Model.Build().
func MSE(output, target *G.Node) *G.Node {
	return G.Must(G.Mean(G.Must(G.Square(G.Must(G.Sub(output, target))))))
}
