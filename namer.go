package goras

import "fmt"

// NewNamer creates a new Namer with the given base name.
func NewNamer(baseName string) func() string {
	counter := 0
	return func() string {
		counter++
		return fmt.Sprintf("%s_%d", baseName, counter)
	}
}
