package shape

// Valid reports whether all dimensions are valid.
func (s Shape) Valid() bool {
	return s.vector.Valid()
}
