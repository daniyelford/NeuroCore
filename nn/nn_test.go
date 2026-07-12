package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestLinear(
	t *testing.T,
) {

	layer :=
		NewLinear(
			3,
			2,
		)

	x := *autograd.NewVariable(
		tensor.New(
			shape.New(1, 3),
		),
		false,
	)

	y := layer.Forward(x)

	if y.Data().Shape().Values()[1] != 2 {

		t.Fatal()

	}

}

func TestStateDict(t *testing.T) {

	model :=
		NewModel(
			NewLinear(3, 2),
		)

	state :=
		model.StateDict()

	if len(state) != 2 {

		t.Fatal(
			"wrong state size",
		)

	}

	_, ok :=
		state["weight"]

	if !ok {

		t.Fatal(
			"weight missing",
		)

	}

}
