package goras

func copyMap[T comparable, U any](dst, src map[T]U) {
	for k, v := range src {
		dst[k] = v
	}
}
