package goras

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// A OneHotLayer is a layer that performs a one-hot encoding of the input.
// The input should be a 1D tensor of integers (batchsize,).
// The output will be a 2D tensor of the specified dtype (batchsize, numClasses).
type OneHotLayer struct {
	LayerBase
	NumClasses int
	DType      T.Dtype
}

// Parameters implements Layer.
func (*OneHotLayer) Parameters() map[string]*G.Node {
	return map[string]*G.Node{}
}

func OneHot(m *Model, name string, dtype T.Dtype, numClasses int) *OneHotLayer {
	if numClasses < 1 {
		panic("numClasses must be greater than 0")
	}
	o := &OneHotLayer{LayerBase{m.Graph, name, "onehot", false, nil, nil}, numClasses, dtype}
	m.AddLayer(o)
	return o
}

// Attach attaches the layer to a previous node.
func (l *OneHotLayer) Attach(n *G.Node) (*G.Node, error) {
	if err := validateShape(n.Shape(), valNDims(1)); err != nil {
		return nil, err
	}
	if n.Dtype() != G.Int {
		return nil, fmt.Errorf("OneHotLayer only supports integer inputs")
	}
	output, err := G.ApplyOp(&oneHotOp{numClasses: l.NumClasses, dType: l.DType}, n)
	if err != nil {
		return nil, err
	}
	l.InputNodes = []*G.Node{n}
	l.OutputNode = output
	return output, nil
}

func (l *OneHotLayer) MustAttach(n *G.Node) *G.Node { return mustAttach(l, n) }
