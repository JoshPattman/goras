package goras

import (
	G "gorgonia.org/gorgonia"
)

type Conv2DLayer struct {
	LayerBase
	Kernels    *G.Node
	KernelSize []int
	NumKernels int
	Stride     []int
	Padding    string
}

func SimpleConv2D(m *Model, name string, kernelSize int, numKernels int) *Conv2DLayer {
	l := &Conv2DLayer{
		LayerBase{m.Graph, name, true},
		nil,
		[]int{kernelSize, kernelSize},
		numKernels,
		[]int{kernelSize, kernelSize},
		"same",
	}
	m.AddLayer(l)
	return l
}

func (l *Conv2DLayer) Attach(x *G.Node) (*G.Node, error) {
	if err := validateShape(x.Shape(), valNDims(4)); err != nil {
		return nil, err
	}
	pad := []int{0, 0} // padding=valid
	if l.Padding == "same" {
		pad = []int{l.KernelSize[0] / 2, l.KernelSize[1] / 2}
	}
	previousKernels := x.Shape()[1]
	l.Kernels = G.NewTensor(l.Graph, G.Float64, 4, G.WithShape(l.NumKernels, previousKernels, l.KernelSize[0], l.KernelSize[1]), G.WithInit(G.GlorotN(1.0)))
	return G.Conv2d(x, l.Kernels, l.KernelSize, pad, l.Stride, []int{1, 1})
}

// Attach attaches the layer to a previous node.
// It then returns the node that the layer outputs.
// Panics if there is an error.
func (l *Conv2DLayer) MustAttach(n *G.Node) *G.Node {
	n, err := l.Attach(n)
	if err != nil {
		panic(err)
	}
	return n
}

// Parameters returns a map of the parameters of the layer.
func (l *Conv2DLayer) Parameters() map[string]*G.Node {
	return map[string]*G.Node{"kernels": l.Kernels}
}
