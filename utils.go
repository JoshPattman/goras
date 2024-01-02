package goras

import (
	"reflect"

	"gorgonia.org/tensor"
	T "gorgonia.org/tensor"
)

type nilHelperType int

var nilType = T.Dtype{Type: reflect.TypeOf(nilHelperType(0))}

func copyMap[T comparable, U any](dst, src map[T]U) {
	for k, v := range src {
		dst[k] = v
	}
}

// NamedTs is a map of string to T.Tensor.
// It is just a convenience type to make code nicer to read.
type NamedTs map[string]T.Tensor

// Return a list of all axes of a tensor
func allAxes(shape tensor.Shape) []int {
	axes := make([]int, shape.Dims())
	for i := range axes {
		axes[i] = i
	}
	return axes
}
