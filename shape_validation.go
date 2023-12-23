package goras

import (
	"fmt"

	"gorgonia.org/tensor"
	T "gorgonia.org/tensor"
)

type shapeValidator func(T.Shape) error

func validateShape(shape T.Shape, vals ...shapeValidator) error {
	for _, val := range vals {
		if err := val(shape); err != nil {
			return err
		}
	}
	return nil
}

func valNDims(n int) shapeValidator {
	return func(s T.Shape) error {
		if len(s) != n {
			return fmt.Errorf("expected shape with ndims %v but got ndims %v with shape %v", len(s), n, s)
		}
		return nil
	}
}

func valNthDim(dim int, val int) shapeValidator {
	return func(s T.Shape) error {
		if s[dim] != val {
			return fmt.Errorf("expected shape[%v] to be %v but got %v", dim, val, s[dim])
		}
		return nil
	}
}

func valMatchingDim(target T.Shape) shapeValidator {
	return func(s T.Shape) error {
		if !s.Eq(target) {
			return fmt.Errorf("expected shape %v but got %v", target, s)
		}
		return nil
	}
}

func valMatchingVolume(target T.Shape) shapeValidator {
	return func(s T.Shape) error {
		if s.TotalSize() != target.TotalSize() {
			return fmt.Errorf("shapes must have the same size: %v and %v", target, s)
		}
		return nil
	}
}

func valAtLeastNDims(n int) shapeValidator {
	return func(s T.Shape) error {
		if len(s) < n {
			return fmt.Errorf("expected shape with at least %v dims but got %v", n, len(s))
		}
		return nil
	}
}

func checkBatchedInputShapes(m *Model, inps []tensor.Tensor) error {
	if len(inps) != len(m.InputNodes) {
		return fmt.Errorf("incorrect number of inputs. expected %v but got %v", len(m.InputNodes), len(inps))
	}

	for i := range inps {
		if !exactShapeEq(m.InputNodes[i].Shape(), inps[i].Shape()) {
			return fmt.Errorf("input %v had incorrect shape. expected %v but got %v", i, m.InputNodes[i].Shape(), inps[i].Shape())
		}
	}
	return nil
}

func exactShapeEq(a, b tensor.Shape) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
