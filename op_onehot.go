package goras

// WARNING - I think this should probably be in gorgonia, but for now it will live here.

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var _ gorgonia.Op = &oneHotOp{}
var _ gorgonia.SDOp = &oneHotOp{}

type oneHotOp struct {
	numClasses int
	dType      tensor.Dtype
}

// DiffWRT implements gorgonia.SDOp.
func (*oneHotOp) DiffWRT(inputs int) []bool {
	// I'm pretty sure you cant, nor would ever want to, take the derivative of this op.
	return make([]bool, inputs)
}

// SymDiff implements gorgonia.SDOp.
func (*oneHotOp) SymDiff(inputs gorgonia.Nodes, output *gorgonia.Node, grad *gorgonia.Node) (retVal gorgonia.Nodes, err error) {
	panic("unimplemented (tho tbf this should never be called)")
}

// Arity implements gorgonia.Op.
func (*oneHotOp) Arity() int {
	return 1 // we expect just a vector of indices
}

// CallsExtern implements gorgonia.Op.
func (*oneHotOp) CallsExtern() bool {
	return false
}

// Do implements gorgonia.Op.
func (op *oneHotOp) Do(inp ...gorgonia.Value) (gorgonia.Value, error) {
	batchSize := inp[0].Shape()[0]
	tens := tensor.New(tensor.WithShape(batchSize, op.numClasses), tensor.Of(op.dType))
	for i := 0; i < batchSize; i++ {
		index := inp[0].Data().([]int)[i]
		var err error
		switch op.dType {
		case tensor.Int:
			err = tens.SetAt(int(1), i, index)
		case tensor.Float64:
			err = tens.SetAt(float64(1), i, index)
		case tensor.Float32:
			err = tens.SetAt(float32(1), i, index)
		case tensor.Bool:
			err = tens.SetAt(true, i, index)
		}
		if err != nil {
			return nil, err
		}
	}
	return tens, nil
}

// InferShape implements gorgonia.Op.
func (op *oneHotOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	s = append(s, op.numClasses)
	return s, nil
}

// OverwritesInput implements gorgonia.Op.
func (*oneHotOp) OverwritesInput() int {
	return -1
}

// ReturnsPtr implements gorgonia.Op.
func (*oneHotOp) ReturnsPtr() bool {
	return false
}

// String implements gorgonia.Op.
func (*oneHotOp) String() string {
	return "OneHotOp"
}

// Type implements gorgonia.Op.
func (*oneHotOp) Type() hm.Type {
	ohTypeInput := gorgonia.TensorType{
		Dims: 1,
		Of:   tensor.Int,
	}
	ohTypeOutput := gorgonia.TensorType{
		Dims: 2,
		Of:   tensor.Float64,
	}
	return hm.NewFnType(ohTypeInput, ohTypeOutput)
}

// I dont actually know what this is for (i just copied this code from another op)
func (op *oneHotOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, op.String()) }

// Hashcode implements gorgonia.Op.
func (*oneHotOp) Hashcode() uint32 {
	// I dont actually know what this is for
	panic("unimplementedb")
}
