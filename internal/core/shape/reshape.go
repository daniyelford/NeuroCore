package shape

// CanReshape reports whether the shape can be reshaped into target.
func (s Shape) CanReshape(target Shape) bool {
	return s.NumElements() == target.NumElements()
}
