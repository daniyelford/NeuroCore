package autograd

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd/operations"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestBackwardRoot(t *testing.T) {

	data := tensor.New(
		shape.New(1),
	)

	data.Set(
		2,
		0,
	)

	x := NewVariable(
		data,
		true,
	)

	Backward(x)

	if x.Grad().Len() != 1 {

		t.Fatal(
			"gradient not created",
		)

	}

}
func TestSimpleBackward(
	t *testing.T,
) {

	xData := tensor.New(
		shape.New(1),
	)

	xData.Set(
		2,
		0,
	)

	x := NewVariable(
		xData,
		true,
	)

	y := Add(
		x,
		x,
	)

	Backward(
		y,
	)
	println(
		"x grad len:",
		x.Grad().Len(),
	)
	if x.Grad().At(0) != 2 {

		t.Fatal(
			x.Grad().At(0),
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

	x := NewVariable(
		data,
		true,
	)

	engine := NewEngine()

	y, err :=
		engine.Execute(
			&operations.Add{},
			x,
			x,
		)

	if err != nil {
		t.Fatal(err)
	}

	Backward(y)

	if x.Grad().At(0) != 2 {

		t.Fatal(
			x.Grad().At(0),
		)

	}

}
