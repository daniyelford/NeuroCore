package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestSum(t *testing.T) {

	tensor := From(
		shape.New(3),
		[]float32{
			1, 2, 3,
		},
	)

	if tensor.Sum() != 6 {

		t.Fatal()

	}

}

func TestMean(t *testing.T) {

	tensor := From(
		shape.New(4),
		[]float32{
			2, 4, 6, 8,
		},
	)

	if tensor.Mean() != 5 {

		t.Fatal()

	}

}
