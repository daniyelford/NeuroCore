package stride

func (s Stride) Equal(other Stride) bool {
	return s.vector.Equal(other.vector)
}
