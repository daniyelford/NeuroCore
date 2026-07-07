package shape

// Rank returns the number of dimensions.
func (s Shape) Rank() int {
	return s.vector.Rank()
}

// Len returns the number of dimensions.
func (s Shape) Len() int {
	return s.vector.Len()
}

// At returns the dimension at index i.
func (s Shape) At(i int) int {
	return s.vector.At(i)
}

// Get safely returns the dimension at index i.
func (s Shape) Get(i int) (int, bool) {
	return s.vector.Get(i)
}

// First returns the first dimension.
func (s Shape) First() int {
	return s.vector.First()
}

// Last returns the last dimension.
func (s Shape) Last() int {
	return s.vector.Last()
}

// Values returns a copy of all dimensions.
func (s Shape) Values() []int {
	return s.vector.Values()
}
