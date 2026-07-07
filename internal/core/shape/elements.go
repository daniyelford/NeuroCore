package shape

// NumElements returns the total number of elements.
func (s Shape) NumElements() int {
	if s.Len() == 0 {
		return 0
	}

	product := 1

	s.vector.Range(func(_ int, value int) bool {
		product *= value
		return true
	})

	return product
}
