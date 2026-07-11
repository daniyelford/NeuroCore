package dataset

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestDataLoader(t *testing.T) {

	x :=
		tensor.New(
			shape.New(10, 3),
		)

	y :=
		tensor.New(
			shape.New(10, 1),
		)

	ds :=
		NewTensorDataset(x, y)

	loader :=
		NewDataLoader(
			ds,
			4,
		)

	count := 0

	for batch := range loader.Batches() {

		if batch.X.Shape().Values()[0] > 4 {

			t.Fatal()

		}

		count++

	}

	if count != 3 {

		t.Fatal()

	}

}
