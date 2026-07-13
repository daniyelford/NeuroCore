package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestPad2D(t *testing.T) {

	x :=
		New(
			shape.New(2, 2),
		)

	x.Set(1, 0, 0)
	x.Set(2, 0, 1)
	x.Set(3, 1, 0)
	x.Set(4, 1, 1)

	out, ok :=
		x.Pad(
			1,
			1,
			1,
			1,
			0,
		)

	if !ok {
		t.Fatal("pad failed")
	}

	values :=
		out.Shape().Values()

	if values[0] != 4 ||
		values[1] != 4 {

		t.Fatal("wrong pad shape")

	}

	if out.At(1, 1) != 1 {
		t.Fatal("wrong copied value")
	}

	if out.At(2, 2) != 4 {
		t.Fatal("wrong copied center")
	}

	if out.At(0, 0) != 0 {
		t.Fatal("wrong padding value")
	}

}
