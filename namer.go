package goras

import "fmt"

// Namer is helper type to generate names for layers.
// It is initialised with a base name (for example "model"),
// and then it generates names in the format "baseName_counter" (eg "model_1", "model_2").
type Namer struct {
	BaseName string
	Counter  int
}

// NewNamer creates a new Namer with the given base name.
func NewNamer(baseName string) *Namer {
	return &Namer{BaseName: baseName}
}

// Next generates the next name.
func (n *Namer) Next() string {
	n.Counter++
	return n.BaseName + "_" + fmt.Sprint(n.Counter)
}
