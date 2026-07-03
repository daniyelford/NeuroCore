package scalar

func (s *Scalar[T]) Add(other *Scalar[T]) *Scalar[T] {

	return &Scalar[T]{
		value: s.value + other.value,
	}

}

func (s *Scalar[T]) Sub(other *Scalar[T]) *Scalar[T] {

	return &Scalar[T]{
		value: s.value - other.value,
	}

}

func (s *Scalar[T]) Mul(other *Scalar[T]) *Scalar[T] {

	return &Scalar[T]{
		value: s.value * other.value,
	}

}

func (s *Scalar[T]) Div(other *Scalar[T]) *Scalar[T] {

	return &Scalar[T]{
		value: s.value / other.value,
	}

}