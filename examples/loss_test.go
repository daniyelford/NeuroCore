package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

func TestMSE(t *testing.T) {

	p := tensor.From(
		shape.New(2),
		[]float32{
			1, 2,
		},
	)

	tg := tensor.From(
		shape.New(2),
		[]float32{
			2, 4,
		},
	)

	l := nn.NewMSE()

	out := l.Forward(
		autograd.NewVariable(p, false),
		autograd.NewVariable(tg, false),
	)

	if out.Data().At(0) != 2.5 {

		t.Fatal()

	}

}
