package goras

import (
	"math"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

type MaxPooling2DLayer struct {
	LayerBase
	PoolSize []int
	Stride   []int
	Padding  string
}

// SimpleMaxPooling2D creates a new max pooling layer on the specified model.
// It will have padding=same stride=poolSize, and it is the same in both dims.
// This means if the input has dims (x,y), and pooling is applied with pool size p,
// then the output will have dims (x/p, y/p) IF x and y are divisible by p.
// Input shape must be BCHW (batch, channels, height, width)
func SimpleMaxPooling2D(m *Model, name string, poolSize int) *MaxPooling2DLayer {
	l := &MaxPooling2DLayer{
		LayerBase{m.Graph, name, false, m.DType},
		[]int{poolSize, poolSize},
		[]int{poolSize, poolSize},
		"same",
	}
	m.AddLayer(l)
	return l
}

// MaxPooling2D creates a new max pooling layer on the specified model.
// Padding can be either "same" or "valid".
func MaxPooling2D(m *Model, name string, poolSize, stride []int, padding string) *MaxPooling2DLayer {
	l := &MaxPooling2DLayer{
		LayerBase{m.Graph, name, false, m.DType},
		poolSize,
		stride,
		padding,
	}
	m.AddLayer(l)
	return l
}

func (l *MaxPooling2DLayer) Attach(x *G.Node) (*G.Node, error) {
	if err := validateShape(x.Shape(), valNDims(4)); err != nil {
		return nil, err
	}
	pad := []int{0, 0} // padding=valid
	if l.Padding == "same" {
		padH := calculateSamePadding(x.Shape()[2], l.PoolSize[0], l.Stride[0])
		padW := calculateSamePadding(x.Shape()[3], l.PoolSize[1], l.Stride[1])
		pad = append(padH, padW...)
	}
	return G.MaxPool2D(x, T.Shape(l.PoolSize), pad, l.Stride)
}

func (l *MaxPooling2DLayer) MustAttach(x *G.Node) *G.Node {
	n, err := l.Attach(x)
	if err != nil {
		panic(err)
	}
	return n
}

// Parameters returns a map of the parameters of the layer.
func (l *MaxPooling2DLayer) Parameters() map[string]*G.Node { return map[string]*G.Node{} }

// I borrowed the calculations from here: https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow/
func calculateSamePadding(width, filterSize, stride int) []int {
	outWidth := int(math.Ceil(float64(width) / float64(stride)))
	padAlongWidth := int(math.Max(float64((outWidth-1)*stride+filterSize-width), 0))
	padLeft := padAlongWidth / 2
	padRight := padAlongWidth - padLeft
	return []int{padLeft, padRight}
}

func (l *MaxPooling2DLayer) Type() string { return "maxpool2d" }
