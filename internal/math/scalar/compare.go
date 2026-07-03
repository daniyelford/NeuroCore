package scalar

func (s *Scalar[T]) Equal(other *Scalar[T]) bool {

	return s.value == other.value

}