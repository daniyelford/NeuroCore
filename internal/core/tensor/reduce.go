package tensor

import "github.com/daniyelford/neurocore/internal/core/shape"

func (t Tensor) SumTensor() Tensor {

	sum := t.Sum()

	out := New(
		shape.New(1),
	)

	out.Set(sum, 0)

	return out

}
func (t Tensor) Sum() float32 {

	var result float32

	for i := 0; i < t.Len(); i++ {

		result += t.memory.At(i)

	}

	return result

}
func (t Tensor) Mean() float32 {

	if t.Len() == 0 {

		return 0

	}

	return t.Sum() / float32(t.Len())

}
func (t Tensor) Min() float32 {

	if t.Len() == 0 {

		return 0

	}

	value := t.memory.At(0)

	for i := 1; i < t.Len(); i++ {

		if t.memory.At(i) < value {

			value = t.memory.At(i)

		}

	}

	return value

}
func (t Tensor) Max() float32 {

	if t.Len() == 0 {

		return 0

	}

	value := t.memory.At(0)

	for i := 1; i < t.Len(); i++ {

		if t.memory.At(i) > value {

			value = t.memory.At(i)

		}

	}

	return value

}
func (t Tensor) ReduceMean() Tensor {

	value := t.Mean()

	out := New(
		shape.New(1),
	)

	out.Set(
		value,
		0,
	)

	return out

}
