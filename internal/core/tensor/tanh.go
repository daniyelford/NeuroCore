package tensor

import "math"

func (t Tensor) Tanh() Tensor {

	out := t.Clone()

	for i := 0; i < out.NumElements(); i++ {

		idx := out.memoryIndex(i)

		v := out.memory.At(idx)

		out.memory.Set(
			idx,
			float32(
				math.Tanh(
					float64(v),
				),
			),
		)

	}

	return out

}
func (t Tensor) TanhBackward(
	grad Tensor,
) Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.NumElements(); i++ {

		idx := t.memoryIndex(i)

		v := t.memory.At(idx)

		g := grad.memory.At(
			grad.memoryIndex(i),
		)

		out.memory.Set(
			out.memoryIndex(i),
			g*(1-v*v),
		)

	}

	return out
}
