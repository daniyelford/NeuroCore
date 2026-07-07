package shape

// Clone returns a deep copy of the shape.
func (s Shape) Clone() Shape {
	return Shape{
		vector: s.vector.Clone(),
	}
}
