package goras

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// BinElemmArithmeticLayer is a layer that can perform many types of binary element-wise arithmetic
type BinElemmArithmeticLayer struct {
	LayerBase
	ArithOp string
}

func BinElemArithmetic(m *Model, name string, op string) *BinElemmArithmeticLayer {
	l := &BinElemmArithmeticLayer{
		LayerBase{
			m.Graph, name, "arith(" + op + ")", false, nil, nil,
		},
		op,
	}
	m.AddLayer(l)
	return l
}

func Add(m *Model, name string) *BinElemmArithmeticLayer {
	return BinElemArithmetic(m, name, "add")
}

func Sub(m *Model, name string) *BinElemmArithmeticLayer {
	return BinElemArithmetic(m, name, "sub")
}

func HardmanProd(m *Model, name string) *BinElemmArithmeticLayer {
	return BinElemArithmetic(m, name, "hardman_prod")
}

func HardmanDiv(m *Model, name string) *BinElemmArithmeticLayer {
	return BinElemArithmetic(m, name, "hardman_div")
}

func Dot(m *Model, name string) *BinElemmArithmeticLayer {
	return BinElemArithmetic(m, name, "dot")
}

// TODO: add shape checking for dot
func (l *BinElemmArithmeticLayer) Attach(a, b *G.Node) (*G.Node, error) {
	var out *G.Node
	var err error
	switch l.ArithOp {
	case "add":
		out, err = G.Add(a, b)
	case "sub":
		out, err = G.Sub(a, b)
	case "hardman_prod":
		out, err = G.HadamardProd(a, b)
	case "hardman_div":
		out, err = G.HadamardDiv(a, b)
	case "dot":
		mulled, err2 := G.HadamardProd(a, b)
		if err2 != nil {
			err = err2
			break
		}
		out, err = G.Sum(mulled, 1)
	default:
		return nil, fmt.Errorf("invalid arith op '%s'", l.ArithOp)
	}
	l.OutputNode = out
	G.WithName(l.Name() + ".op")(out)
	l.InputNodes = []*G.Node{a, b}
	return out, err
}

// Parameters implements Layer.
func (*BinElemmArithmeticLayer) Parameters() map[string]*G.Node {
	return make(map[string]*G.Node)
}
