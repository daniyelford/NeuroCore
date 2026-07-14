package tensor

import (
	"math"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

// ------------------------------------------------
// Scale
// ------------------------------------------------
func Scalar(
	value float32,
) Tensor {

	out := New(
		shape.New(1),
	)

	out.FlatSet(
		0,
		value,
	)

	return out

}
func (t Tensor) Scale(
	value float32,
) Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.NumElements(); i++ {

		idx := t.memoryIndex(i)

		out.memory.Set(
			idx,
			t.memory.At(idx)*value,
		)

	}

	return out

}

//------------------------------------------------
// Scalar helpers
//------------------------------------------------

func (t Tensor) ScalarMul(
	value float32,
) Tensor {

	return t.Scale(value)

}

func (t Tensor) DivScalar(
	value float32,
) Tensor {

	return t.Scale(
		1 / value,
	)

}

func (t Tensor) AddScalar(
	value float32,
) Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.NumElements(); i++ {

		idx := t.memoryIndex(i)

		out.memory.Set(
			idx,
			t.memory.At(idx)+value,
		)

	}

	return out

}

func (t Tensor) SubScalar(
	value float32,
) Tensor {

	return t.AddScalar(
		-value,
	)

}

func (t Tensor) Neg() Tensor {

	return t.Scale(
		-1,
	)

}

//------------------------------------------------
// Dot
//------------------------------------------------

func (t Tensor) Dot(
	other Tensor,
) (float32, bool) {

	if t.NumElements() != other.NumElements() {

		return 0,
			false

	}

	var sum float32

	for i := 0; i < t.NumElements(); i++ {

		idxA :=
			t.memoryIndex(i)

		idxB :=
			other.memoryIndex(i)

		sum +=
			t.memory.At(idxA) *
				other.memory.At(idxB)

	}

	return sum,
		true

}

//------------------------------------------------
// Clone
//------------------------------------------------

func (t Tensor) ScalarClone() Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.NumElements(); i++ {

		idx :=
			t.memoryIndex(i)

		out.memory.Set(
			idx,
			t.memory.At(idx),
		)

	}

	return out

}

//------------------------------------------------
// Compare
//------------------------------------------------

func (t Tensor) ScalarEqual(
	other Tensor,
) bool {

	if !t.Shape().Equal(
		other.Shape(),
	) {

		return false

	}

	for i := 0; i < t.NumElements(); i++ {

		if t.FlatAt(i) !=
			other.FlatAt(i) {

			return false

		}

	}

	return true

}

func (t Tensor) AllClose(
	other Tensor,
	eps float32,
) bool {

	if !t.Shape().Equal(
		other.Shape(),
	) {

		return false

	}

	for i := 0; i < t.NumElements(); i++ {

		diff :=
			float32(
				math.Abs(
					float64(
						t.FlatAt(i) -
							other.FlatAt(i),
					),
				),
			)

		if diff > eps {

			return false

		}

	}

	return true

}
