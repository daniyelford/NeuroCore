package optim_test

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/optim"
)

func TestSGDStep(
	t *testing.T,
) {

	data :=
		tensor.New(
			shape.New(1),
		)

	data.Set(
		10,
		0,
	)

	variable :=
		autograd.NewVariable(
			data,
			true,
		)

	parameter :=
		nn.NewParameter(
			variable,
		)

	grad :=
		tensor.New(
			shape.New(1),
		)

	grad.Set(
		2,
		0,
	)

	variable.SetGrad(
		grad,
	)

	optimizer :=
		optim.NewSGD(
			[]nn.Parameter{
				parameter,
			},
			0.1,
		)

	optimizer.Step()

	result :=
		variable.Data().FlatAt(0)

	expected :=
		float32(9.8)

	if result != expected {

		t.Fatalf(
			"expected %v got %v",
			expected,
			result,
		)

	}

}
