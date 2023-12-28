// The stuff in this file is just some stuff to make working with tensors easier.
// There is a pretty good chance that this stuff is already in gorgonia, but I couldn't find it after all 2 seconds of looking I did.
package goras

import (
	"fmt"

	"gorgonia.org/tensor"
)

// Make2DSliceTensor converts a 2D slice to a tensor. The slice is indexed[row][column].
func Make2DSliceTensor[T any](data [][]T) (tensor.Tensor, error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil, fmt.Errorf("data slice must have at least one row and one column")
	}
	var eg T
	typ, err := GetTensorDataType(eg)
	if err != nil {
		return nil, err
	}
	t := tensor.New(tensor.WithShape(len(data), len(data[0])), tensor.Of(typ))
	for i, row := range data {
		if len(row) != len(data[0]) {
			return nil, fmt.Errorf("data slice must have the same number of columns in each row")
		}
		for j, v := range row {
			if err := t.SetAt(v, i, j); err != nil {
				return nil, err
			}
		}
	}
	return t, nil
}

// Make1DSliceTensor converts a 1D slice to a tensor.
func Make1DSliceTensor[T any](data []T) (tensor.Tensor, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data slice must have at least one element")
	}
	var eg T
	typ, err := GetTensorDataType(eg)
	if err != nil {
		return nil, err
	}
	t := tensor.New(tensor.WithShape(len(data)), tensor.Of(typ))
	for i, v := range data {
		if err := t.SetAt(v, i); err != nil {
			return nil, err
		}
	}
	return t, nil
}

func GetTensorDataType(t interface{}) (tensor.Dtype, error) {
	switch t.(type) {
	case int:
		return tensor.Int, nil
	case float64:
		return tensor.Float64, nil
	case float32:
		return tensor.Float32, nil
	case bool:
		return tensor.Bool, nil
	default:
		return tensor.Dtype{}, fmt.Errorf("unsupported type %T", t)
	}
}
