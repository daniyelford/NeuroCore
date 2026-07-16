package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type MSE struct {
	Base
}

func (op *MSE) Name() string {

	return "MSELoss"

}
func (op *MSE) Forward(
	inputs ...*autograd.Variable,
) (
	*autograd.Variable,
	error,
) {

	if len(inputs) != 2 {

		return nil,
			errors.New(
				"mse loss requires prediction and target",
			)

	}

	prediction := inputs[0]

	target := inputs[1]

	op.Save(
		prediction,
		target,
	)

	diff :=
		prediction.Data().Sub(
			target.Data(),
		)

	squared :=
		diff.Mul(
			diff,
		)

	loss :=
		squared.ReduceMean()

	out :=
		autograd.NewVariable(
			loss,
			prediction.RequiresGrad(),
		)

	out.Node().Parents =
		[]*autograd.Node{

			prediction.Node(),

			target.Node(),
		}

	out.Node().Op = op

	op.SetOutput(
		out,
	)

	return out,
		nil

}
func (op *MSE) Backward(
	grad tensor.Tensor,
) (
	[]tensor.Tensor,
	error,
) {

	pred :=
		op.Input(0).Data()

	target :=
		op.Input(1).Data()

	diff :=
		pred.Sub(
			target,
		)

	scale :=
		float32(
			2.0 /
				float32(
					diff.NumElements(),
				),
		)

	dPred :=
		diff.MulScalar(
			scale,
		)

	dPred =
		dPred.ScalarMul(
			grad.FlatAt(0),
		)
		// dPred =
	// 	dPred.Mul(
	// 		grad,
	// 	)

	dTarget :=
		dPred.Neg()

	return []tensor.Tensor{

		dPred,

		dTarget,
	}, nil

}
