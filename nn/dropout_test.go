package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestDropoutParameters(t *testing.T) {

	d := NewDropout(0.5)

	if len(d.Parameters()) != 0 {

		t.Fatal("dropout should not have parameters")

	}

}

func TestDropoutStateDict(t *testing.T) {

	d := NewDropout(0.5)

	if len(d.StateDict()) != 0 {

		t.Fatal("dropout should not have state")

	}

}

func TestDropoutEval(t *testing.T) {

	x := tensor.New(
		shape.New(4),
	)

	x.FlatSet(0, 1)
	x.FlatSet(1, 2)
	x.FlatSet(2, 3)
	x.FlatSet(3, 4)

	v := autograd.NewVariable(
		x,
		false,
	)

	d := NewDropout(0.5)

	d.Eval()

	out := d.Forward(*v)

	for i := 0; i < x.Len(); i++ {

		if out.Data().FlatAt(i) != x.FlatAt(i) {

			t.Fatal("eval mode should not modify tensor")

		}

	}

}

func TestDropoutTrain(t *testing.T) {

	x := tensor.New(
		shape.New(100),
	)

	x.Fill(1)

	v := autograd.NewVariable(
		x,
		false,
	)

	d := NewDropout(0.5)

	d.Train()

	out := d.Forward(*v)

	hasZero := false

	for i := 0; i < out.Data().Len(); i++ {

		if out.Data().FlatAt(i) == 0 {

			hasZero = true
			break

		}

	}

	if !hasZero {

		t.Fatal("expected at least one dropped element")

	}

}
