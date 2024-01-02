package goras

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

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
