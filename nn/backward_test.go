package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/internal/operations"
)

func TestBackwardRoot(t *testing.T) {

	data := tensor.New(
		shape.New(1),
	)

	data.Set(
		2,
		0,
	)

	x := autograd.NewVariable(
		data,
		true,
	)

	autograd.Backward(x)

	if x.Grad().Len() != 1 {

		t.Fatal(
			"gradient not created",
		)

	}

}
func TestSimpleBackwardEngin(t *testing.T) {

	data := tensor.New(
		shape.New(1),
	)

	data.Set(
		2,
		0,
	)

	x := autograd.NewVariable(
		data,
		true,
	)
	engine := autograd.NewEngine()

	y, err :=
		engine.Execute(
			new(operations.Add),
			x,
			x,
		)
	println(x, y, err)

	if err != nil {
		t.Fatal(err)
	}

	autograd.Backward(y)

	if x.Grad().At(0) != 2 {

		t.Fatal(
			x.Grad().At(0),
		)

	}

}
