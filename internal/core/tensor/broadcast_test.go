package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestBroadcastScalar(t *testing.T) {

	a := From(
		shape.New(3),
		[]float32{
			1, 2, 3,
		},
	)

	b := From(
		shape.New(1),
		[]float32{
			10,
		},
	)

	c, ok := a.AddBroadcast(b)

	if !ok {

		t.Fatal()

	}

	if c.At(0) != 11 {

		t.Fatal()

	}

	if c.At(2) != 13 {

		t.Fatal()

	}

}
