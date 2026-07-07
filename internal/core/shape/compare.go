package shape

// Equal reports whether two shapes are identical.
func (s Shape) Equal(other Shape) bool {
	return s.vector.Equal(other.vector)
}
