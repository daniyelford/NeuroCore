package tensor

import (
	"fmt"
	"math"

	"github.com/daniyelford/neurocore/internal/core/backend"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/stride"
)

func binaryOp(a Tensor, b Tensor, op func(float32, float32) float32) Tensor {
	out, ok := broadcastBinary(a, b, op)
	if !ok {
		panic("shape mismatch")
	}
	return out
}
func (t Tensor) Add(other Tensor) Tensor {
	return binaryOp(t, other, func(a, b float32) float32 { return a + b })
}
func (t Tensor) Sub(other Tensor) Tensor {

	return binaryOp(
		t,
		other,

		func(
			a,
			b float32,
		) float32 {

			return a - b

		},
	)

}
func (t Tensor) Mul(
	other Tensor,
) Tensor {

	return binaryOp(
		t,
		other,

		func(
			a,
			b float32,
		) float32 {

			return a * b

		},
	)

}

func (t Tensor) Div(
	other Tensor,
) Tensor {

	return binaryOp(
		t,
		other,

		func(
			a,
			b float32,
		) float32 {

			return a / b

		},
	)

}
func (t Tensor) At(
	indices ...int,
) float32 {
	index :=
		t.offset +
			t.stride.Offset(indices...)

	return t.memory.At(index)

}
func (t Tensor) Set(
	value float32,
	indices ...int,
) {

	index :=
		t.offset +
			t.stride.Offset(indices...)

	t.memory.Set(
		index,
		value,
	)

}
func (t Tensor) TryAt(
	indices ...int,
) (float32, bool) {

	offset, ok := t.stride.TryOffset(indices...)

	if !ok {
		return 0, false
	}

	return t.memory.TryAt(offset)

}
func (t Tensor) TrySet(
	value float32,
	indices ...int,
) bool {

	offset, ok := t.stride.TryOffset(indices...)

	if !ok {
		return false
	}

	return t.memory.TrySet(
		offset,
		value,
	)

}
func (t Tensor) memoryIndex(
	linear int,
) int {

	return t.offset + linear

}
func (t Tensor) FlatAt(
	index int,
) float32 {

	return t.memory.At(
		t.offset + index,
	)

}
func (t Tensor) FlatSet(
	index int,
	value float32,
) {

	t.memory.Set(
		t.offset+index,
		value,
	)

}
func (t Tensor) ArgMax() int {

	maxIndex := 0

	maxValue :=
		t.FlatAt(0)

	for i := 1; i < t.NumElements(); i++ {

		v :=
			t.FlatAt(i)

		if v > maxValue {

			maxValue = v

			maxIndex = i

		}

	}

	return maxIndex

}
func (t Tensor) MulScalar(
	value float32,
) Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.Len(); i++ {

		out.FlatSet(
			i,
			t.FlatAt(i)*value,
		)

	}

	return out

}
func OneHot(
	index int,
	classes int,
) Tensor {

	out :=
		New(
			shape.New(classes),
		)

	out.FlatSet(
		index,
		1,
	)

	return out

}
func (t Tensor) Pad(
	top int,
	bottom int,
	left int,
	right int,
	value float32,
) (Tensor, bool) {

	dims :=
		t.Shape().Values()

	if len(dims) != 2 {
		return Tensor{}, false
	}

	h := dims[0]
	w := dims[1]

	out :=
		New(
			shape.New(
				h+top+bottom,
				w+left+right,
			),
		)

	out.Fill(value)

	for i := 0; i < h; i++ {

		for j := 0; j < w; j++ {

			out.Set(
				t.At(i, j),
				i+top,
				j+left,
			)

		}

	}

	return out, true
}
func (t Tensor) Permute(
	dims ...int,
) (Tensor, bool) {

	old := t.Shape().Values()

	if len(dims) != len(old) {
		return Tensor{}, false
	}

	used := make([]bool, len(old))

	newShape := make([]int, len(dims))

	for i, d := range dims {

		if d < 0 || d >= len(old) {
			return Tensor{}, false
		}

		if used[d] {
			return Tensor{}, false
		}

		used[d] = true

		newShape[i] = old[d]

	}

	out :=
		New(
			shape.New(newShape...),
		)

	for i := 0; i < t.Len(); i++ {

		// فعلا copy ساده
		out.FlatSet(
			i,
			t.FlatAt(i),
		)

	}

	return out, true
}

func (t Tensor) Shape() shape.Shape {

	return t.shape

}

func (t Tensor) Stride() stride.Stride {

	return t.stride

}

func (t Tensor) Device() backend.DeviceType {

	return t.device

}

func (t Tensor) Len() int {

	return t.shape.NumElements()

}
func (t Tensor) NumElements() int {

	return t.shape.NumElements()

}

func (t Tensor) Empty() bool {

	return t.memory.Empty()

}
func (t Tensor) Offset() int {

	return t.offset

}
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
func (t Tensor) ReduceMax() float32 {

	max :=
		float32(
			math.Inf(-1),
		)

	for i := 0; i < t.NumElements(); i++ {

		if v := t.FlatAt(i); v > max {

			max = v

		}

	}

	return max

}
func (t Tensor) ReLU() Tensor {

	out := t.Clone()

	for i := 0; i < out.NumElements(); i++ {

		idx := out.memoryIndex(i)

		v := out.memory.At(idx)

		if v < 0 {

			out.memory.Set(idx, 0)

		}

	}

	return out

}
func (t Tensor) ReLUMask() Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.NumElements(); i++ {

		idx := t.memoryIndex(i)

		v := t.memory.At(idx)

		if v > 0 {

			out.memory.Set(
				out.memoryIndex(i),
				1,
			)

		} else {

			out.memory.Set(
				out.memoryIndex(i),
				0,
			)

		}

	}

	return out
}
func (t Tensor) Reshape(
	newShape shape.Shape,
) (Tensor, bool) {

	if t.NumElements() != newShape.NumElements() {

		return Tensor{}, false

	}

	return Tensor{

		shape: newShape,

		stride: stride.Compute(
			newShape,
			t.layout,
		),

		memory: t.memory,
		offset: t.offset,
		device: t.device,

		layout: t.layout,
	}, true

}
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
func (t Tensor) LogSoftmax() Tensor {

	return t.LogSoftmaxDim(
		0,
	)

}
func (t Tensor) LogSoftmaxDim(
	axis int,
) Tensor {

	dims :=
		t.Shape().Values()

	if len(dims) != 2 || axis != 1 {

		return t.LogSoftmax()

	}

	rows :=
		dims[0]

	cols :=
		dims[1]

	out :=
		New(
			t.Shape(),
		)

	for r := 0; r < rows; r++ {

		// max row
		max :=
			t.At(
				r,
				0,
			)

		for c := 1; c < cols; c++ {

			v :=
				t.At(
					r,
					c,
				)

			if v > max {

				max = v

			}

		}

		// sum(exp(x-max))
		sum :=
			float32(0)

		for c := 0; c < cols; c++ {

			v :=
				t.At(
					r,
					c,
				)

			sum +=
				float32(
					math.Exp(
						float64(
							v - max,
						),
					),
				)

		}

		logSum :=
			float32(
				math.Log(
					float64(sum),
				),
			)

		// x - max - log(sum(exp(x-max)))
		for c := 0; c < cols; c++ {

			value :=
				t.At(
					r,
					c,
				)

			out.Set(
				value-max-logSum,
				r,
				c,
			)

		}

	}

	return out

}
func (t Tensor) Log() Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.NumElements(); i++ {

		idx := t.memoryIndex(i)

		out.memory.Set(
			idx,
			float32(
				math.Log(
					float64(
						t.memory.At(idx),
					),
				),
			),
		)

	}

	return out

}
func (t Tensor) MatMul(
	other Tensor,
) (Tensor, bool) {

	a := t.shape.Values()

	b := other.shape.Values()

	if len(a) != 2 || len(b) != 2 {

		return Tensor{}, false

	}

	if a[1] != b[0] {

		return Tensor{}, false

	}

	out := New(
		shape.New(
			a[0],
			b[1],
		),
	)

	for i := 0; i < a[0]; i++ {

		for j := 0; j < b[1]; j++ {

			var sum float32

			for k := 0; k < a[1]; k++ {

				av := t.At(i, k)

				bv := other.At(k, j)

				sum += av * bv

			}

			out.Set(
				sum,
				i,
				j,
			)

		}

	}

	return out, true

}
func (t Tensor) View(
	newShape shape.Shape,
	offset int,
) (Tensor, bool) {

	if offset < 0 {

		return Tensor{}, false

	}

	if offset+newShape.NumElements() >
		t.memory.Len()-t.offset {

		return Tensor{}, false

	}

	return Tensor{

		shape: newShape,

		stride: stride.Compute(
			newShape,
			t.layout,
		),

		memory: t.memory,

		offset: t.offset + offset,

		device: t.device,

		layout: t.layout,
	}, true

}
func (t Tensor) Slice(
	start int,
	end int,
) (Tensor, bool) {

	dims :=
		t.shape.Values()

	if len(dims) == 0 {

		return Tensor{}, false

	}

	if start < 0 ||
		end > dims[0] ||
		start >= end {

		return Tensor{}, false

	}

	newDims :=
		make([]int, len(dims))

	copy(
		newDims,
		dims,
	)

	newDims[0] =
		end - start

	offset :=
		start * t.stride.At(0)

	return t.View(
		shape.New(newDims...),
		offset,
	)

}
func (t Tensor) Clone() Tensor {

	return Tensor{

		shape: t.shape,

		stride: t.stride,

		memory: t.memory.Clone(),

		device: t.device,

		layout: t.layout,
	}

}
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
			g*(1-v*v))
	}
	return out
}
func Stack(tensors []Tensor) (Tensor, bool) {
	if len(tensors) == 0 {
		return Tensor{}, false
	}
	baseShape := tensors[0].Shape().Values()
	resultShape := append([]int{len(tensors)}, baseShape...)
	out := New(shape.New(resultShape...))
	offset := 0
	for _, t := range tensors {
		if !t.Shape().Equal(tensors[0].Shape()) {
			return Tensor{}, false
		}
		for i := 0; i < t.Len(); i++ {
			out.FlatSet(offset+i, t.FlatAt(i))
		}
		offset += t.Len()
	}
	return out, true
}
func (t Tensor) Squeeze() Tensor {
	out := t
	out.shape = t.shape.Squeeze()
	out.stride = stride.FromShape(out.shape)
	return out
}
func (t Tensor) Softmax() Tensor {
	return t.SoftmaxDim(0)
}
func (t Tensor) SoftmaxDim(axis int) Tensor {
	dims := t.Shape().Values()
	if len(dims) != 2 || axis != 1 {
		return t.Softmax()
	}
	rows := dims[0]
	cols := dims[1]
	out := New(t.Shape())
	for r := 0; r < rows; r++ {
		max := t.At(r, 0)
		for c := 1; c < cols; c++ {
			v := t.At(r, c)
			if v > max {
				max = v
			}
		}
		sum := float32(0)
		for c := 0; c < cols; c++ {
			v := t.At(r, c)
			e := float32(math.Exp(float64(v - max)))
			out.Set(e, r, c)
			sum += e
		}
		for c := 0; c < cols; c++ {
			v := out.At(r, c)
			out.Set(v/sum, r, c)
		}
	}
	return out
}
func (t Tensor) Sigmoid() Tensor {
	out := t.Clone()
	for i := 0; i < out.NumElements(); i++ {
		idx := out.memoryIndex(i)
		v := out.memory.At(idx)
		s := float32(1.0 / (1.0 + math.Exp(-float64(v))))
		out.memory.Set(idx, s)
	}
	return out
}
func (t Tensor) SigmoidBackward(grad Tensor) Tensor {
	out := New(t.Shape())
	for i := 0; i < t.NumElements(); i++ {
		idx := t.memoryIndex(i)
		s := t.memory.At(idx)
		g := grad.memory.At(grad.memoryIndex(i))
		out.memory.Set(out.memoryIndex(i), g*s*(1-s))
	}
	return out
}
func (t Tensor) Transpose() (Tensor, bool) {
	d := t.shape.Values()
	if len(d) != 2 {
		return Tensor{}, false
	}
	out := New(shape.New(d[1], d[0]))
	for i := 0; i < d[0]; i++ {
		for j := 0; j < d[1]; j++ {
			out.Set(t.At(i, j), j, i)
		}
	}
	return out, true
}
func SameShape(a Tensor, b Tensor) bool {
	return a.Shape().Equal(b.Shape())
}
func SameSize(a Tensor, b Tensor) bool {
	return a.Len() == b.Len()
}
func (t Tensor) Valid() bool {
	if !t.shape.Valid() {
		return false
	}
	if !t.stride.Valid() {
		return false
	}
	return t.shape.NumElements() == t.memory.Len()
}
func (t Tensor) Indices() ([]int, error) {
	out := make([]int, t.Len())
	for i := 0; i < t.Len(); i++ {
		v := t.FlatAt(i)
		index := int(v)
		if float32(index) != v {
			return nil, fmt.Errorf("tensor value %v is not an integer index", v)
		}
		out[i] = index
	}
	return out, nil
}
func (t Tensor) AddInplace(other Tensor) bool {
	if !t.shape.Equal(other.shape) {
		return false
	}
	for i := 0; i < t.Len(); i++ {
		t.memory.Set(i, t.memory.At(i)+other.memory.At(i))
	}
	return true
}
func (t Tensor) SubInplace(other Tensor) bool {
	if !t.shape.Equal(other.shape) {
		return false
	}
	for i := 0; i < t.Len(); i++ {
		t.memory.Set(i, t.memory.At(i)-other.memory.At(i))
	}
	return true
}
func (t Tensor) ScaleInplace(value float32) {
	for i := 0; i < t.Len(); i++ {
		t.memory.Set(i, t.memory.At(i)*value)
	}
}
func (t Tensor) Flatten() (Tensor, bool) {
	return t.Reshape(shape.New(t.NumElements()))
}
func (t Tensor) Fill(value float32) {
	t.memory.Fill(value)
}
func (t Tensor) Zero() {
	t.Fill(0)
}
func (t Tensor) Exp() Tensor {
	out := New(t.Shape())
	for i := 0; i < t.NumElements(); i++ {
		idx := t.memoryIndex(i)
		out.memory.Set(idx, float32(math.Exp(float64(t.memory.At(idx)))))
	}
	return out
}
func (t Tensor) Equal(other Tensor) bool {
	return t.shape.Equal(other.shape) && t.memory.Equal(other.memory)
}
func broadcastShape(a shape.Shape, b shape.Shape) (shape.Shape, bool) {
	ad := a.Values()
	bd := b.Values()
	max := len(ad)
	if len(bd) > max {
		max = len(bd)
	}
	result := make([]int, max)
	for i := 0; i < max; i++ {
		av := 1
		bv := 1
		if len(ad)-1-i >= 0 {
			av = ad[len(ad)-1-i]
		}
		if len(bd)-1-i >= 0 {
			bv = bd[len(bd)-1-i]
		}
		if av != bv && av != 1 && bv != 1 {
			return shape.Shape{}, false
		}
		if av > bv {
			result[max-1-i] = av
		} else {
			result[max-1-i] = bv
		}
	}
	return shape.New(result...), true
}
func (t Tensor) broadcastAt(index int, target shape.Shape) float32 {
	targetIndices := linearToIndices(index, target)
	srcIndices := broadcastIndices(targetIndices, t.Shape())
	return t.At(srcIndices...)
}
func (t Tensor) Broadcast(target shape.Shape) (Tensor, bool) {
	_, ok := broadcastShape(t.Shape(), target)
	if !ok {
		return Tensor{}, false
	}
	out := New(target)
	for i := 0; i < out.NumElements(); i++ {
		out.FlatSet(i, t.broadcastAt(i, target))
	}
	return out, true
}
func (t Tensor) AddBroadcast(other Tensor) (Tensor, bool) {
	return broadcastBinary(t, other, func(a, b float32) float32 { return a + b })
}
func linearToIndices(index int, sh shape.Shape) []int {
	dims := sh.Values()
	indices := make([]int, len(dims))
	for i := len(dims) - 1; i >= 0; i-- {
		indices[i] = index % dims[i]
		index /= dims[i]
	}
	return indices
}
func indicesToLinear(indices []int, sh shape.Shape) int {
	dims := sh.Values()
	stride := 1
	index := 0
	for i := len(dims) - 1; i >= 0; i-- {
		index += indices[i] * stride
		stride *= dims[i]
	}
	return index
}
func broadcastIndices(targetIndices []int, src shape.Shape) []int {
	srcDims := src.Values()
	out := make([]int, len(srcDims))
	offset := len(targetIndices) - len(srcDims)
	for i := range srcDims {
		j := i + offset
		if j < 0 {
			out[i] = 0
			continue
		}
		if srcDims[i] == 1 {
			out[i] = 0
		} else {
			out[i] = targetIndices[j]
		}
	}
	return out
}
func (t Tensor) SubBroadcast(other Tensor) (Tensor, bool) {
	return broadcastBinary(t, other, func(a, b float32) float32 { return a - b })
}
func (t Tensor) MulBroadcast(other Tensor) (Tensor, bool) {
	return broadcastBinary(t, other, func(a, b float32) float32 { return a * b })
}
func (t Tensor) DivBroadcast(other Tensor) (Tensor, bool) {
	return broadcastBinary(t, other, func(a, b float32) float32 { return a / b })
}
func broadcastBinary(a Tensor, b Tensor, op func(float32, float32) float32) (Tensor, bool) {
	outShape, ok := broadcastShape(a.Shape(), b.Shape())
	if !ok {
		return Tensor{}, false
	}
	out := New(outShape)
	for i := 0; i < out.NumElements(); i++ {
		av := a.broadcastAt(i, outShape)
		bv := b.broadcastAt(i, outShape)
		out.FlatSet(i, op(av, bv))
	}
	return out, true
}
