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

func MSELoss(targetName string, output *G.Node) LossFunc {
	return func() (*G.Node, map[string]*G.Node, error) {
		target := G.NewMatrix(output.Graph(), G.Float64, G.WithShape(1, 1))
		x, err := G.Sub(output, target)
		if err != nil {
			return nil, nil, err
		}
		x, err = G.Square(x)
		if err != nil {
			return nil, nil, err
		}
		x, err = G.Mean(x)
		if err != nil {
			return nil, nil, err
		}
		return x, map[string]*G.Node{targetName: target}, nil
	}
}

// MSE creates the nodes to calculate mean squared error loss between a predicted and target node.
// It should be used when using Model.Build().
func MSE(output, target *G.Node) (*G.Node, error) {
	x, err := G.Sub(output, target)
	if err != nil {
		return nil, err
	}
	x, err = G.Square(x)
	if err != nil {
		return nil, err
	}
	x, err = G.Mean(x)
	if err != nil {
		return nil, err
	}
	return x, nil
}

// BCE creates the nodes to calculate binary crossentropy loss between a predicted and target node.
// It should be used when using Model.Build().
func BCE(output, target *G.Node) (*G.Node, error) {
	x1, err := G.Log(output)
	if err != nil {
		return nil, err
	}
	x2, err := G.Sub(G.NewConstant(1.0), output)
	if err != nil {
		return nil, err
	}
	x2, err = G.Log(x2)
	if err != nil {
		return nil, err
	}
	x1, err = G.HadamardProd(target, x1)
	if err != nil {
		return nil, err
	}
	x3, err := G.Sub(G.NewConstant(1.0), target)
	if err != nil {
		return nil, err
	}
	x2, err = G.HadamardProd(x3, x2)
	if err != nil {
		return nil, err
	}
	x, err := G.Add(x1, x2)
	if err != nil {
		return nil, err
	}
	x, err = G.Mean(x)
	if err != nil {
		return nil, err
	}
	x, err = G.Neg(x)
	if err != nil {
		return nil, err
	}
	return x, nil
}

// CCE creates the nodes to calculate categorical crossentropy loss between a predicted and target node.
// It should be used when using Model.Build().
// IMPORTANT: I think this is currently broken. Either this or softmax.
func CCE(output, target *G.Node) (*G.Node, error) {
	x, err := G.Log(output)
	if err != nil {
		return nil, err
	}
	x, err = G.HadamardProd(target, x)
	if err != nil {
		return nil, err
	}
	x, err = G.Sum(x, 1)
	if err != nil {
		return nil, err
	}
	x, err = G.Mean(x)
	if err != nil {
		return nil, err
	}
	x, err = G.Neg(x)
	if err != nil {
		return nil, err
	}
	return x, nil
}
