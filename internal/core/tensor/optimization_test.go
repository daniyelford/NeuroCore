package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestIterator(t *testing.T) {

	tensor := From(
		shape.New(3),
		[]float32{
			1, 2, 3,
		},
	)

	it := tensor.Iterator()

	sum := float32(0)

	for it.HasNext() {

		sum += it.Next()

	}

	if sum != 6 {

		t.Fatal()

	}

}
