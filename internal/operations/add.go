package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Add struct {
	Base
}

var _ autograd.Operation = (*Add)(nil)

func (op *Add) Name() string {

	return "Add"

}
func (op *Add) Forward(
	inputs ...*autograd.Variable,
) (
	*autograd.Variable,
	error,
) {

	if len(inputs) != 2 {

		return nil,
			errors.New(
				"add requires 2 inputs",
			)

	}

	a := inputs[0]

	b := inputs[1]

	op.Save(
		a,
		b,
	)

	out :=
		autograd.NewVariable(
			a.Data().Add(
				b.Data(),
			),
			a.RequiresGrad() ||
				b.RequiresGrad(),
		)

	op.SetOutput(out)

	return out, nil

}
func (op *Add) Backward(
	grad tensor.Tensor,
) (
	[]tensor.Tensor,
	error,
) {

	return []tensor.Tensor{

		grad,

		grad,
	}, nil

}
