package init

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestZeros(t *testing.T) {

	x :=
		tensor.New(
			shape.New(2, 3),
		)

	Zeros{}.Init(&x)

	for i := 0; i < x.Len(); i++ {

		if x.FlatAt(i) != 0 {
			t.Fatal("zeros initializer failed")
		}

	}

}

func TestOnes(t *testing.T) {

	x :=
		tensor.New(
			shape.New(2, 3),
		)

	Ones{}.Init(&x)

	for i := 0; i < x.Len(); i++ {

		if x.FlatAt(i) != 1 {
			t.Fatal("ones initializer failed")
		}

	}

}

func TestUniform(t *testing.T) {

	x :=
		tensor.New(
			shape.New(10, 10),
		)

	Uniform{
		Min: -1,
		Max: 1,
	}.Init(&x)

	for i := 0; i < x.Len(); i++ {

		v :=
			x.FlatAt(i)

		if v < -1 || v > 1 {
			t.Fatal("uniform value out of range")
		}

	}

}

func TestNormal(t *testing.T) {

	x :=
		tensor.New(
			shape.New(10, 10),
		)

	Normal{
		Mean: 0,
		Std:  1,
	}.Init(&x)

	allZero := true

	for i := 0; i < x.Len(); i++ {

		if x.FlatAt(i) != 0 {
			allZero = false
			break
		}

	}

	if allZero {
		t.Fatal("normal initializer produced all zeros")
	}

}

func TestXavier(t *testing.T) {

	x :=
		tensor.New(
			shape.New(4, 8),
		)

	Xavier{}.Init(&x)

	allZero := true

	for i := 0; i < x.Len(); i++ {

		if x.FlatAt(i) != 0 {

			allZero = false
			break

		}

	}

	if allZero {

		t.Fatal("xavier initializer failed")

	}

}

func TestKaiming(t *testing.T) {

	x :=
		tensor.New(
			shape.New(4, 8),
		)

	Kaiming{}.Init(&x)

	allZero := true

	for i := 0; i < x.Len(); i++ {

		if x.FlatAt(i) != 0 {

			allZero = false
			break

		}

	}

	if allZero {

		t.Fatal("kaiming initializer failed")

	}

}
