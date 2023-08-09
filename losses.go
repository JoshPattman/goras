package goras

import (
	G "gorgonia.org/gorgonia"
)

// MSE creates the nodes to calculate mean squared error loss between a predicted and target node.
// It should be used when using Model.Build().
func MSE(output, target *G.Node) *G.Node {
	return G.Must(G.Mean(G.Must(G.Square(G.Must(G.Sub(output, target))))))
}

// BCE creates the nodes to calculate binary crossentropy loss between a predicted and target node.
// It should be used when using Model.Build().
func BCE(output, target *G.Node) *G.Node {
	logPi := G.Must(G.Log(output))
	logOneMinusPi := G.Must(G.Log(G.Must(G.Sub(G.NewConstant(1.0), output))))
	firstTerm := G.Must(G.HadamardProd(target, logPi))
	secondTerm := G.Must(G.HadamardProd(G.Must(G.Sub(G.NewConstant(1.0), target)), logOneMinusPi))
	return G.Must(G.Neg(G.Must(G.Mean(G.Must(G.Add(firstTerm, secondTerm))))))
}

// CCE creates the nodes to calculate categorical crossentropy loss between a predicted and target node.
// It should be used when using Model.Build().
func CCE(output, target *G.Node) *G.Node {
	logPi := G.Must(G.Log(output))
	mulProb := G.Must(G.HadamardProd(target, logPi))
	summed := G.Must(G.Sum(mulProb, 1))
	return G.Must(G.Neg(G.Must(G.Mean(summed))))
}
