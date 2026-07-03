package scalar

func (s *Scalar[T]) Value() T {
	return s.value
}

func (s *Scalar[T]) Set(v T) {
	s.value = v
}

func (s *Scalar[T]) Clone() *Scalar[T] {
	return &Scalar[T]{
		value: s.value,
	}
}