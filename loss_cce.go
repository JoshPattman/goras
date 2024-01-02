package goras

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

func CCELoss(targetName string, output *G.Node) LossFunc {
	return func() (*G.Node, map[string]*G.Node, error) {
		target := G.NewMatrix(output.Graph(), output.Dtype(), G.WithShape(output.Shape()...), G.WithName(targetName))
		x, err := G.Log(output)
		if err != nil {
			return nil, nil, fmt.Errorf("CCE error while performing Log op: %v", err)
		}
		x, err = G.HadamardProd(target, x)
		if err != nil {
			return nil, nil, fmt.Errorf("CCE error while performing HardmanProd op: %v", err)
		}
		x, err = G.Sum(x, 1)
		if err != nil {
			return nil, nil, fmt.Errorf("CCE error while performing Sum op: %v", err)
		}
		x, err = G.Mean(x)
		if err != nil {
			return nil, nil, fmt.Errorf("CCE error while performing Mean op: %v", err)
		}
		x, err = G.Neg(x)
		if err != nil {
			return nil, nil, fmt.Errorf("CCE error while performing Neg op: %v", err)
		}
		return x, map[string]*G.Node{targetName: target}, nil
	}
}
