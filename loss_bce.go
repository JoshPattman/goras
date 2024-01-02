package goras

import G "gorgonia.org/gorgonia"

// BCE creates the nodes to calculate binary crossentropy loss between a predicted and target node.
// It should be used when using Model.Build().
func BCELoss(targetName string, output *G.Node) LossFunc {
	return func() (*G.Node, map[string]*G.Node, error) {
		target := G.NewMatrix(output.Graph(), output.Dtype(), G.WithShape(output.Shape()...), G.WithName(targetName))
		x1, err := G.Log(output)
		if err != nil {
			return nil, nil, err
		}
		x2, err := G.Sub(G.NewConstant(1.0, G.WithName(targetName+".const1a")), output)
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
		x3, err := G.Sub(G.NewConstant(1.0, G.WithName(targetName+".const1b")), target)
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
