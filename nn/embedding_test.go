package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestEmbeddingShape(t *testing.T) {

	e :=
		NewEmbedding(
			100,
			8,
		)

	s :=
		e.Weight.Value.Data().
			Shape().
			Values()

	if s[0] != 100 ||
		s[1] != 8 {

		t.Fatal(
			"wrong embedding shape",
		)

	}

}

func TestEmbeddingForward(t *testing.T) {

	input :=
		tensor.New(
			shape.New(3),
		)

	input.Set(1, 0)
	input.Set(5, 1)
	input.Set(9, 2)

	v :=
		autograd.NewVariable(
			input,
			false,
		)

	e :=
		NewEmbedding(
			20,
			4,
		)

	out :=
		e.Forward(
			*v,
		)

	s :=
		out.Data().
			Shape().
			Values()

	if s[0] != 3 ||
		s[1] != 4 {

		t.Fatal(
			"invalid embedding output shape",
		)

	}

}

func TestEmbeddingLookup(t *testing.T) {

	e :=
		NewEmbedding(
			10,
			3,
		)

	w :=
		e.Weight.Value.Data()

	w.Set(10, 2, 0)
	w.Set(20, 2, 1)
	w.Set(30, 2, 2)

	input :=
		tensor.New(
			shape.New(1),
		)

	input.Set(
		2,
		0,
	)

	out :=
		e.Forward(
			*autograd.NewVariable(
				input,
				false,
			),
		)

	if out.Data().At(0, 0) != 10 ||
		out.Data().At(0, 1) != 20 ||
		out.Data().At(0, 2) != 30 {

		t.Fatal(
			"lookup failed",
		)

	}

}
