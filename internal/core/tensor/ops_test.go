package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestAdd(t *testing.T) {

	a := From(
		shape.New(3),
		[]float32{1, 2, 3},
	)

	b := From(
		shape.New(3),
		[]float32{4, 5, 6},
	)

	c := a.Add(b)

	if c.At(0) != 5 {

		t.Fatal()

	}

	if c.At(2) != 9 {

		t.Fatal()

	}

}
func TestAddInplace(t *testing.T) {

	a := From(
		shape.New(3),
		[]float32{1, 2, 3},
	)

	b := From(
		shape.New(3),
		[]float32{4, 5, 6},
	)

	ok := a.AddInplace(b)

	if !ok {

		t.Fatal()

	}

	if a.At(0) != 5 {

		t.Fatal()

	}

}
