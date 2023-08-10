package goras

import (
	"fmt"

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
