package autograd

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestAddBackward(
	t *testing.T,
) {

	a :=
		NewVariable(
			tensor.From(
				shape.New(1),
				[]float32{2},
			),
			true,
		)

	b :=
		NewVariable(
			tensor.From(
				shape.New(1),
				[]float32{3},
			),
			true,
		)
	// engine := NewEngine()

	// c, err :=
	// 	engine.Execute(
	// 		&Add{},
	// 		a,
	// 		b,
	// 	)

	// Backward(c)

	if a.Grad().At(0) != 1 {

		t.Fatal()

	}

	if b.Grad().At(0) != 1 {

		t.Fatal()

	}

}
