package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

func Accumulate(
	v *Variable,
	grad tensor.Tensor,
) {

	if v.Grad.Empty() {

		v.Grad = grad

		return

	}

	v.Grad =
		v.Grad.Add(
			grad,
		)

}
