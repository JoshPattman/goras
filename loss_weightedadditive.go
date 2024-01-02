package goras

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// KNOWN BUG: I'm pretty certain this will not work if the graph is using float32s, because all the weights are float64
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
			// BUG: the name here is not unique
			scaleNode := G.NewConstant(weights[i], G.WithName(fmt.Sprintf("weightedadditiveloss.weight%d", i)))
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
