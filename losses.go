package goras

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// LossFunc is a function that when called, returns:
//
// - a node (loss output scalar)
//
// - a map of nodes which the loss requires to be created (for instance, this is usually the target for the output layer)
//
// - an error
type LossFunc func() (lossOut *G.Node, lossInps map[string]*G.Node, err error)

// MSE creates the nodes to calculate mean squared error loss between a predicted and target node.
// It should be used when using Model.Build().
func MSELoss(targetName string, output *G.Node) LossFunc {
	return func() (*G.Node, map[string]*G.Node, error) {
		target := G.NewMatrix(output.Graph(), G.Float64, G.WithShape(output.Shape()...))
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

// BCE creates the nodes to calculate binary crossentropy loss between a predicted and target node.
// It should be used when using Model.Build().
func BCELoss(targetName string, output *G.Node) LossFunc {
	return func() (*G.Node, map[string]*G.Node, error) {
		target := G.NewMatrix(output.Graph(), G.Float64, G.WithShape(output.Shape()...))
		x1, err := G.Log(output)
		if err != nil {
			return nil, nil, err
		}
		x2, err := G.Sub(G.NewConstant(1.0), output)
		if err != nil {
			return nil, nil, err
		}
		x2, err = G.Log(x2)
		if err != nil {
			return nil, nil, err
		}
		x1, err = G.HadamardProd(target, x1)
		if err != nil {
			return nil, nil, err
		}
		x3, err := G.Sub(G.NewConstant(1.0), target)
		if err != nil {
			return nil, nil, err
		}
		x2, err = G.HadamardProd(x3, x2)
		if err != nil {
			return nil, nil, err
		}
		x, err := G.Add(x1, x2)
		if err != nil {
			return nil, nil, err
		}
		x, err = G.Mean(x)
		if err != nil {
			return nil, nil, err
		}
		x, err = G.Neg(x)
		if err != nil {
			return nil, nil, err
		}
		return x, map[string]*G.Node{targetName: target}, nil
	}
}

func CCELoss(targetName string, output *G.Node) LossFunc {
	return func() (*G.Node, map[string]*G.Node, error) {
		target := G.NewMatrix(output.Graph(), G.Float64, G.WithShape(output.Shape()...))
		x, err := G.Log(output)
		if err != nil {
			return nil, nil, err
		}
		x, err = G.HadamardProd(target, x)
		if err != nil {
			return nil, nil, err
		}
		x, err = G.Sum(x, 1)
		if err != nil {
			return nil, nil, err
		}
		x, err = G.Mean(x)
		if err != nil {
			return nil, nil, err
		}
		x, err = G.Neg(x)
		if err != nil {
			return nil, nil, err
		}
		return x, map[string]*G.Node{targetName: target}, nil
	}
}

func L2Loss(layers ...Layer) LossFunc {
	return func() (*G.Node, map[string]*G.Node, error) {
		if len(layers) == 0 {
			return nil, nil, fmt.Errorf("no layers provided to L2Loss")
		}
		// get a list of all trainable parameters
		var params []*G.Node
		for _, layer := range layers {
			for _, param := range layer.Parameters() {
				params = append(params, param)
			}
		}
		var sumNodes []*G.Node
		for _, param := range params {
			x, err := G.Square(param)
			if err != nil {
				return nil, nil, err
			}
			x, err = G.Sum(x, allAxes(param.Shape())...)
			if err != nil {
				return nil, nil, err
			}
			sumNodes = append(sumNodes, x)
		}
		total := sumNodes[0]
		for _, node := range sumNodes[1:] {
			var err error
			total, err = G.Add(total, node)
			if err != nil {
				return nil, nil, err
			}
		}
		return total, map[string]*G.Node{}, nil
	}
}

func WeightedAdditiveLoss(losses []LossFunc, weights []float64) LossFunc {
	return func() (*G.Node, map[string]*G.Node, error) {
		if len(losses) != len(weights) {
			return nil, nil, fmt.Errorf("number of losses and weights must match")
		}
		lossNodes := []*G.Node{}
		allLossInps := map[string]*G.Node{}
		for _, loss := range losses {
			lossNode, lossInp, err := loss()
			if err != nil {
				return nil, nil, err
			}
			lossNodes = append(lossNodes, lossNode)
			for k, v := range lossInp {
				if _, ok := allLossInps[k]; ok {
					return nil, nil, fmt.Errorf("loss with name %s already exists", k)
				}
				allLossInps[k] = v
			}
		}
		var total *G.Node
		for i, lossNode := range lossNodes {
			scaleNode := G.NewConstant(weights[i])
			x, err := G.Mul(lossNode, scaleNode)
			if err != nil {
				return nil, nil, err
			}
			if total == nil {
				total = x
			} else {
				total, err = G.Add(total, x)
				if err != nil {
					return nil, nil, err
				}
			}
		}
		return total, allLossInps, nil
	}
}

func allAxes(shape tensor.Shape) []int {
	axes := make([]int, shape.Dims())
	for i := range axes {
		axes[i] = i
	}
	return axes
}
