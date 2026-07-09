package autograd

func (v *Variable) Detach() *Variable {

	return NewVariable(
		v.Data(),
		false,
	)
}
