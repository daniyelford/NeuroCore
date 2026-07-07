package stride

// IsEmpty reports whether the stride has no dimensions.
func (s Stride) IsEmpty() bool {
	return s.Len() == 0
}

// IsScalar reports whether the stride belongs to a scalar.
func (s Stride) IsScalar() bool {
	return s.Len() == 1 && s.At(0) == 1
}
