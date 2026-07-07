package shape

// IsScalar reports whether the shape represents a scalar.
func (s Shape) IsScalar() bool {
	return s.Len() == 1 && s.At(0) == 1
}

// IsVector reports whether the shape represents a vector.
func (s Shape) IsVector() bool {
	return s.Len() == 1 && s.At(0) > 1
}

// IsMatrix reports whether the shape represents a matrix.
func (s Shape) IsMatrix() bool {
	return s.Len() == 2
}

// IsTensor reports whether the shape has three or more dimensions.
func (s Shape) IsTensor() bool {
	return s.Len() >= 3
}
