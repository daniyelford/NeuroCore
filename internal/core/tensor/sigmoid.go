package tensor

import "math"

func (t Tensor) Sigmoid() Tensor {

	out := t.Clone()

	for i := 0; i < out.NumElements(); i++ {

		idx := out.memoryIndex(i)

		v := out.memory.At(idx)

		s := float32(
			1.0 / (1.0 + math.Exp(-float64(v))),
		)

		out.memory.Set(
			idx,
			s,
		)

	}

	return out

}
func (t Tensor) SigmoidBackward(
	grad Tensor,
) Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.NumElements(); i++ {

		idx := t.memoryIndex(i)

		s := t.memory.At(idx)

		g := grad.memory.At(
			grad.memoryIndex(i),
		)

		out.memory.Set(
			out.memoryIndex(i),
			g*s*(1-s),
		)

	}

	return out
}
