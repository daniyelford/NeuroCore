package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestCreate(t *testing.T) {

	sh := shape.New(2, 3)

	tensor := New(sh)

	if tensor.Len() != 6 {

		t.Fatal()

	}

}

func TestSetGet(t *testing.T) {

	tensor := New(
		shape.New(2, 3),
	)

	tensor.Set(
		10,
		1,
		2,
	)

	v := tensor.At(
		1,
		2,
	)

	if v != 10 {

		t.Fatal()

	}

}
func TestFrom(t *testing.T) {

	tensor := From(
		shape.New(2, 2),
		[]float32{
			1, 2, 3, 4,
		},
	)

	if tensor.At(1, 1) != 4 {

		t.Fatal()

	}

}
func TestFill(t *testing.T) {

	tensor := New(
		shape.New(3, 3),
	)

	tensor.Fill(5)

	if tensor.At(2, 2) != 5 {

		t.Fatal()

	}

}
func TestValid(t *testing.T) {

	tensor := New(
		shape.New(2, 3),
	)

	if !tensor.Valid() {

		t.Fatal()

	}

}
func TestTryAccess(t *testing.T) {

	tensor := New(
		shape.New(2, 2),
	)

	ok := tensor.TrySet(
		10,
		1,
		1,
	)

	if !ok {

		t.Fatal()

	}

	v, ok := tensor.TryAt(1, 1)

	if !ok || v != 10 {

		t.Fatal()

	}

	_, ok = tensor.TryAt(10, 10)

	if ok {

		t.Fatal()

	}

}
func TestReshape(t *testing.T) {

	tensor := From(
		shape.New(2, 3),
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
	)

	n, ok := tensor.Reshape(
		shape.New(3, 2),
	)

	if !ok {

		t.Fatal()

	}

	if n.At(2, 1) != 6 {

		t.Fatal()

	}

}
