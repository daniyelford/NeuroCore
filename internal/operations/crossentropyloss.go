package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type CrossEntropyLoss struct {
	Base
}

func (op *CrossEntropyLoss) Name() string {

	return "CrossEntropyLoss"

}

func (op *CrossEntropyLoss) Forward(
	inputs ...*autograd.Variable,
) (
	*autograd.Variable,
	error,
) {

	if len(inputs) != 2 {

		return nil,
			errors.New(
				"cross entropy requires prediction and target",
			)

	}

	logits :=
		inputs[0]

	target :=
		inputs[1]

	op.Save(
		logits,
		target,
	)

	logProb :=
		logits.Data().LogSoftmaxDim(1)

	dims :=
		logProb.Shape().Values()

	batch :=
		dims[0]

	loss :=
		float32(0)

	for i := 0; i < batch; i++ {

		class :=
			int(
				target.Data().At(i),
			)

		loss -=
			logProb.At(
				i,
				class,
			)

	}

	loss /=
		float32(batch)

	outTensor :=
		tensor.Scalar(
			loss,
		)

	out :=
		autograd.NewVariable(
			outTensor,
			logits.RequiresGrad(),
		)

	out.Node().Parents =
		[]*autograd.Node{

			logits.Node(),
		}

	out.Node().Op =
		op

	op.SetOutput(
		out,
	)

	return out,
		nil

}

func (op *CrossEntropyLoss) Backward(
	grad tensor.Tensor,
) (
	[]tensor.Tensor,
	error,
) {

	logits :=
		op.Input(0).Data()

	target :=
		op.Input(1).Data()

	dims :=
		logits.Shape().Values()

	batch :=
		dims[0]

	classes :=
		dims[1]

	prob := logits.SoftmaxDim(1)

	gradInput :=
		tensor.New(
			logits.Shape(),
		)

	for i := 0; i < batch; i++ {

		class :=
			int(
				target.At(i),
			)

		for c := 0; c < classes; c++ {

			value :=
				prob.At(
					i,
					c,
				)

			if c == class {

				value -= 1

			}

			value /=
				float32(batch)

			gradInput.Set(
				value,
				i,
				c,
			)

		}

	}

	gradValue :=
		grad.FlatAt(0)

	gradInput =
		gradInput.ScalarMul(
			gradValue,
		)

	return []tensor.Tensor{

		gradInput,
	}, nil

}
