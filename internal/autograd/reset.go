package autograd

func (v *Variable) ZeroGrad() {

	if v.node.Grad.Empty() {
		return
	}

	v.node.Grad.Zero()

}
