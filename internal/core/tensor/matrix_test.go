package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestMatMul(t *testing.T) {

	a := From(
		shape.New(2, 3),
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
	)

	b := From(
		shape.New(3, 2),
		[]float32{
			7, 8,
			9, 10,
			11, 12,
		},
	)

	c, ok := a.MatMul(b)

	if !ok {

		t.Fatal()

	}

	if c.At(0, 0) != 58 {

		t.Fatal()

	}

	if c.At(1, 1) != 154 {

		t.Fatal()

	}

}
func TestTranspose(t *testing.T) {

	a := From(
		shape.New(2, 3),
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
	)

	b, ok := a.Transpose()

	if !ok {

		t.Fatal()

	}

	if b.At(0, 1) != 4 {

		t.Fatal()

	}

}
