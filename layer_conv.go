package goras

import (
	G "gorgonia.org/gorgonia"
)

// Conv2DLayer is a 2D convolutional layer.
//   - Input Shape: (batch_size, previous_kernels/previous_channels, img_width, img_height)
//   - Output Shape: (batch_size, num_kernels, img_width, img_height)
type Conv2DLayer struct {
	LayerBase
	Kernels    *G.Node
	KernelSize []int
	NumKernels int
	Stride     []int
	Padding    string
}

// SimpleConv2D is a constructor to create a 2D convolutional layer.
// It has a kernel shape of [kernelSize, kernelSize], a stride of [kernelSize, kernelSize], and padding of "same".
func SimpleConv2D(m *Model, name string, kernelSize int, numKernels int) *Conv2DLayer {
	l := &Conv2DLayer{
		LayerBase{m.Graph, name, "conv2d", true, m.DType, nil},
		nil,
		[]int{kernelSize, kernelSize},
		numKernels,
		[]int{1, 1},
		"same",
	}
	m.AddLayer(l)
	return l
}

// Conv2D is a constructor to create a 2D convolutional layer.
// Options for padding are "same" or "valid".
func Conv2D(m *Model, name string, kernelShape, stride []int, padding string, numKernels int) *Conv2DLayer {
	l := &Conv2DLayer{
		LayerBase{m.Graph, name, "conv2d", true, m.DType, nil},
		nil,
		kernelShape,
		numKernels,
		stride,
		padding,
	}
	m.AddLayer(l)
	return l
}

// Attach attaches this layer to a previous node.
func (l *Conv2DLayer) Attach(x *G.Node) (*G.Node, error) {
	if err := validateShape(x.Shape(), valNDims(4)); err != nil {
		return nil, err
	}
	pad := []int{0, 0} // padding=valid
	if l.Padding == "same" {
		pad = []int{l.KernelSize[0] / 2, l.KernelSize[1] / 2}
	}
	previousKernels := x.Shape()[1]
	l.Kernels = G.NewTensor(l.Graph, l.DType, 4, G.WithShape(l.NumKernels, previousKernels, l.KernelSize[0], l.KernelSize[1]), G.WithInit(G.GlorotN(1.0)))
	on, err := G.Conv2d(x, l.Kernels, l.KernelSize, pad, l.Stride, []int{1, 1})
	l.OutputNode = on
	if on != nil {
		G.WithName(l.Name() + ".conv")(on)
	}
	return on, err
}

// MustAttach attaches this layer to a previous node and panics if there is an error.
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
