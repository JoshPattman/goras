package goras

import T "gorgonia.org/tensor"

type ReshapeLayer struct {
	LayerBase
	ToShape T.Shape
}

func Reshape(model Model, name string, newShape T.Shape) *ReshapeLayer {
	return nil
}
