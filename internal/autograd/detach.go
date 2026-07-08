package autograd

func (v Variable) Detach() Variable {

	return Variable{

		Data: v.Data,

		RequiresGrad: false,
	}

}
