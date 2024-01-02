package goras

import G "gorgonia.org/gorgonia"

// MSE creates the nodes to calculate mean squared error loss between a predicted and target node.
// It should be used when using Model.Build().
func MSELoss(targetName string, output *G.Node) LossFunc {
	return func() (*G.Node, map[string]*G.Node, error) {
		target := G.NewMatrix(output.Graph(), output.Dtype(), G.WithShape(output.Shape()...), G.WithName(targetName))
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
