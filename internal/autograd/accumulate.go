package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

func Accumulate(
	v *Variable,
	grad tensor.Tensor,
) {

	current := v.Grad()

	if current.Empty() {

		v.SetGrad(
			grad,
		)

		return
	}

	v.SetGrad(
		current.Add(
			grad,
		),
	)

}
