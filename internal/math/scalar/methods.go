package scalar

func (s *Scalar[T]) Value() T {
	return s.value
}

func (s *Scalar[T]) Gradient() T {
	return s.grad
}

func (s *Scalar[T]) RequiresGrad() bool {
	return s.requiresGrad
}

func (s *Scalar[T]) EnableGrad() {
	s.requiresGrad = true
}

func (s *Scalar[T]) DisableGrad() {
	s.requiresGrad = false
}

func (s *Scalar[T]) Clone() *Scalar[T] {

	return &Scalar[T]{
		value:        s.value,
		grad:         s.grad,
		requiresGrad: s.requiresGrad,
	}

}