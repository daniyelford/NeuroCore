package nn_test

import (
	"testing"

	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/nn/conv"
)

func TestSequentialConvParameters(t *testing.T) {

	model :=
		nn.NewSequential(
			conv.NewConv2D(
				1,
				4,
				3,
				3,
			),
			nn.NewLinear(
				4,
				1,
			),
		)

	params :=
		model.Parameters()

	if len(params) != 4 {

		t.Fatalf(
			"expected 4 parameters got %d",
			len(params),
		)

	}

}

func TestSequentialStateDict(t *testing.T) {

	model :=
		nn.NewSequential(
			conv.NewConv2D(
				1,
				2,
				3,
				3,
			),
		)

	state :=
		model.StateDict()

	expected :=
		[]string{
			"0.weight",
			"0.bias",
		}

	for _, key := range expected {

		if _, ok := state[key]; !ok {

			t.Fatalf(
				"missing state key %s",
				key,
			)

		}

	}

}
