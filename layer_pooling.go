package goras

import (
	"math"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// MaxPooling2DLayer is a max pooling layer.
//   - Input Shape: (batch_size, num_channels, img_height, img_width)
//   - Output Shape: (batch_size, num_channels, img_height, img_width) [img_height and img_width will be smaller than the input]
type MaxPooling2DLayer struct {
	LayerBase
	PoolSize []int
	Stride   []int
	Padding  string
}

// SimpleMaxPooling2D creates a new max pooling layer on the specified model.
// It will have padding=same stride=poolSize, and it is the same in both dims.
func SimpleMaxPooling2D(m *Model, name string, poolSize int) *MaxPooling2DLayer {
	l := &MaxPooling2DLayer{
		LayerBase{m.Graph, name, "maxpool2d", false, m.DType, nil, nil},
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
		LayerBase{m.Graph, name, "maxpool2d", false, m.DType, nil, nil},
		poolSize,
		stride,
		padding,
	}
	m.AddLayer(l)
	return l
}

// Attach attaches the MaxPooling2DLayer to the given node.
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
	on, err := G.MaxPool2D(x, T.Shape(l.PoolSize), pad, l.Stride)
	l.OutputNode = on
	if on != nil {
		G.WithName(l.Name() + ".maxpool")(on)
	}
	l.InputNodes = []*G.Node{x}
	return on, err
}

// MustAttach attaches the MaxPooling2DLayer to the given node.
func (l *MaxPooling2DLayer) MustAttach(n *G.Node) *G.Node { return mustAttach(l, n) }

// Parameters returns a map of the parameters of the layer.
func (l *MaxPooling2DLayer) Parameters() map[string]*G.Node { return map[string]*G.Node{} }

// This function calculates the padding for "same".
// I borrowed the calculations from here: https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow/
func calculateSamePadding(width, filterSize, stride int) []int {
	outWidth := int(math.Ceil(float64(width) / float64(stride)))
	padAlongWidth := int(math.Max(float64((outWidth-1)*stride+filterSize-width), 0))
	padLeft := padAlongWidth / 2
	padRight := padAlongWidth - padLeft
	return []int{padLeft, padRight}
}
