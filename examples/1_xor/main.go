package main

import (
	//G "gorgonia.org/gorgonia"
	"fmt"

	T "gorgonia.org/tensor"
	//K "github.com/JoshPattman/goras"
)

func main() {
	x, y := LoadXY()

}

func LoadXY() (*T.Dense, *T.Dense) {
	x := T.New(
		T.WithShape(4, 2),
		T.WithBacking([]float64{0, 0, 0, 1, 1, 0, 1, 1}),
	)
	y := T.New(
		T.WithShape(4, 1),
		T.WithBacking([]float64{0, 1, 1, 0}),
	)
	fmt.Printf("X:\n%v\n", x)
	fmt.Printf("Y:\n%v\n", y)
	return x, y
}
