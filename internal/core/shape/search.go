package shape

// Dimensions returns a copy of all dimensions.
func (s Shape) Dimensions() []int {
	return s.Values()
}

// Contains reports whether the shape contains dim.
func (s Shape) Contains(dim int) bool {
	return s.vector.Contains(dim)
}

// IndexOf returns the index of dim.
func (s Shape) IndexOf(dim int) int {
	return s.vector.IndexOf(dim)
}
