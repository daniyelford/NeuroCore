package autograd

func (v *Variable) ZeroGrad() {

	v.Grad.Zero()

}
