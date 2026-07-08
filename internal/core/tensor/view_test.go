package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestViewShareMemory(t *testing.T) {

	tensor := From(
		shape.New(5),
		[]float32{
			1, 2, 3, 4, 5,
		},
	)

	v, ok := tensor.View(
		shape.New(3),
		1,
	)

	if !ok {

		t.Fatal()

	}

	if v.At(0) != 2 {

		t.Fatal()

	}

	v.Set(
		100,
		0,
	)

	if tensor.At(1) != 100 {

		t.Fatal("view copied memory")

	}

}
