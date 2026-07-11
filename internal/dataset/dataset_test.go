package dataset

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestTensorDataset(t *testing.T) {

	x :=
		tensor.New(
			shape.New(10, 3),
		)

	y :=
		tensor.New(
			shape.New(10, 1),
		)

	ds :=
		NewTensorDataset(
			x,
			y,
		)

	if ds.Len() != 10 {

		t.Fatal()

	}

	x1, y1 :=
		ds.Get(0)

	if x1.Shape().Values()[0] != 1 {

		t.Fatal()

	}

	if y1.Shape().Values()[0] != 1 {

		t.Fatal()

	}

}
