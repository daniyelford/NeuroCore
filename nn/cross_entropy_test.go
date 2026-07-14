package nn_test

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

func TestCrossEntropyLoss(
	t *testing.T,
) {

	logitsTensor :=
		tensor.New(
			shape.New(
				2,
				3,
			),
		)

	// sample 0
	logitsTensor.Set(
		2,
		0,
		0,
	)

	logitsTensor.Set(
		1,
		0,
		1,
	)

	logitsTensor.Set(
		0,
		0,
		2,
	)

	// sample 1
	logitsTensor.Set(
		0,
		1,
		0,
	)

	logitsTensor.Set(
		1,
		1,
		1,
	)

	logitsTensor.Set(
		2,
		1,
		2,
	)

	logits :=
		autograd.NewVariable(
			logitsTensor,
			true,
		)

	targetTensor :=
		tensor.New(
			shape.New(2),
		)

	targetTensor.Set(
		0,
		0,
	)

	targetTensor.Set(
		2,
		1,
	)

	target :=
		autograd.NewVariable(
			targetTensor,
			false,
		)

	lossFn :=
		nn.NewCrossEntropyLoss()

	out :=
		lossFn.Forward(
			*logits,
			*target,
		)

	if out.Data().NumElements() != 1 {

		t.Fatalf(
			"expected scalar loss",
		)

	}

	autograd.Backward(
		&out,
	)

	if logits.Grad().NumElements() != 6 {

		t.Fatalf(
			"invalid gradient shape",
		)

	}

}
