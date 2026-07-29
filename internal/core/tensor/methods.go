/*
Package tensor defines multidimensional tensors.
*/
package tensor

import (
	"fmt"
	"math"

	"github.com/daniyelford/neurocore/internal/core/backend"
	"github.com/daniyelford/neurocore/internal/core/layout"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/stride"
	"github.com/daniyelford/neurocore/internal/memory"
)

func New(sh shape.Shape) Tensor {
	size := sh.NumElements()
	return Tensor{shape: sh, stride: stride.Compute(sh, layout.RowMajor), memory: memory.New(size), device: backend.CPU, layout: layout.RowMajor}
}
func From(sh shape.Shape, values []float32) Tensor {
	t := New(sh)
	t.memory.CopyFrom(values)
	return t
}
func (t Tensor) Iterator() Iterator {
	return Iterator{tensor: t, index: 0}
}
func (it *Iterator) HasNext() bool {
	return it.index < it.tensor.Len()
}
func (it *Iterator) Next() float32 {
	v := it.tensor.memory.At(it.index)
	it.index++
	return v
}
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
	return binaryOp(t, other, func(a, b float32) float32 { return a - b })
}
func (t Tensor) Mul(other Tensor) Tensor {
	return binaryOp(t, other, func(a, b float32) float32 { return a * b })
}
func (t Tensor) Div(other Tensor) Tensor {
	return binaryOp(t, other, func(a, b float32) float32 { return a / b })
}
func (t Tensor) At(indices ...int) float32 {
	index := t.offset + t.stride.Offset(indices...)
	return t.memory.At(index)
}
func (t Tensor) Set(value float32, indices ...int) {
	index := t.offset + t.stride.Offset(indices...)
	t.memory.Set(index, value)
}
func (t Tensor) TryAt(indices ...int) (float32, bool) {
	offset, ok := t.stride.TryOffset(indices...)
	if !ok {
		return 0, false
	}
	return t.memory.TryAt(offset)
}
func (t Tensor) TrySet(value float32, indices ...int) bool {
	offset, ok := t.stride.TryOffset(indices...)
	if !ok {
		return false
	}
	return t.memory.TrySet(offset, value)
}
func (t Tensor) memoryIndex(linear int) int {
	return t.offset + linear
}
func (t Tensor) FlatAt(index int) float32 {
	return t.memory.At(t.offset + index)
}
func (t Tensor) FlatSet(index int, value float32) {
	t.memory.Set(t.offset+index, value)
}
func (t Tensor) ArgMax() int {
	maxIndex := 0
	maxValue := t.FlatAt(0)
	for i := 1; i < t.NumElements(); i++ {
		v := t.FlatAt(i)
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}
	return maxIndex
}
func (t Tensor) MulScalar(value float32) Tensor {
	out := New(t.Shape())
	for i := 0; i < t.Len(); i++ {
		out.FlatSet(i, t.FlatAt(i)*value)
	}
	return out
}
func OneHot(index int, classes int) Tensor {
	out := New(shape.New(classes))
	out.FlatSet(index, 1)
	return out
}
func (t Tensor) Pad(top int, bottom int, left int, right int, value float32) (Tensor, bool) {
	dims := t.Shape().Values()
	if len(dims) != 2 {
		return Tensor{}, false
	}
	h := dims[0]
	w := dims[1]
	out := New(shape.New(h+top+bottom, w+left+right))
	out.Fill(value)
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			out.Set(t.At(i, j), i+top, j+left)
		}
	}
	return out, true
}
func (t Tensor) Permute(dims ...int) (Tensor, bool) {
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
	out := New(shape.New(newShape...))
	for i := 0; i < t.Len(); i++ {
		out.FlatSet(i, t.FlatAt(i))
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
	out := New(shape.New(1))
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
	out := New(shape.New(1))
	out.Set(value, 0)
	return out
}
func (t Tensor) ReduceMax() float32 {
	max := float32(math.Inf(-1))
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
	out := New(t.Shape())
	for i := 0; i < t.NumElements(); i++ {
		idx := t.memoryIndex(i)
		v := t.memory.At(idx)
		if v > 0 {
			out.memory.Set(out.memoryIndex(i), 1)
		} else {
			out.memory.Set(out.memoryIndex(i), 0)
		}
	}
	return out
}
func (t Tensor) Reshape(newShape shape.Shape) (Tensor, bool) {
	if t.NumElements() != newShape.NumElements() {
		return Tensor{}, false
	}
	return Tensor{
		shape:  newShape,
		stride: stride.Compute(newShape, t.layout),
		memory: t.memory,
		offset: t.offset,
		device: t.device,
		layout: t.layout,
	}, true
}
func Scalar(value float32) Tensor {
	out := New(shape.New(1))
	out.FlatSet(0, value)
	return out
}
func (t Tensor) Scale(value float32) Tensor {
	out := New(t.Shape())
	for i := 0; i < t.NumElements(); i++ {
		idx := t.memoryIndex(i)
		out.memory.Set(idx, t.memory.At(idx)*value)
	}
	return out
}
func (t Tensor) ScalarMul(value float32) Tensor {
	return t.Scale(value)
}
func (t Tensor) DivScalar(value float32) Tensor {
	return t.Scale(1 / value)
}
func (t Tensor) AddScalar(value float32) Tensor {
	out := New(t.Shape())
	for i := 0; i < t.NumElements(); i++ {
		idx := t.memoryIndex(i)
		out.memory.Set(idx, t.memory.At(idx)+value)
	}
	return out
}
func (t Tensor) SubScalar(value float32) Tensor {
	return t.AddScalar(-value)
}
func (t Tensor) Neg() Tensor {
	return t.Scale(-1)
}
func (t Tensor) Dot(other Tensor) (float32, bool) {
	if t.NumElements() != other.NumElements() {
		return 0, false
	}
	var sum float32
	for i := 0; i < t.NumElements(); i++ {
		idxA := t.memoryIndex(i)
		idxB := other.memoryIndex(i)
		sum += t.memory.At(idxA) * other.memory.At(idxB)
	}
	return sum, true
}
func (t Tensor) ScalarClone() Tensor {
	out := New(t.Shape())
	for i := 0; i < t.NumElements(); i++ {
		idx := t.memoryIndex(i)
		out.memory.Set(idx, t.memory.At(idx))
	}
	return out
}
func (t Tensor) ScalarEqual(other Tensor) bool {
	if !t.Shape().Equal(other.Shape()) {
		return false
	}
	for i := 0; i < t.NumElements(); i++ {
		if t.FlatAt(i) != other.FlatAt(i) {
			return false
		}
	}
	return true
}
func (t Tensor) AllClose(other Tensor, eps float32) bool {
	if !t.Shape().Equal(other.Shape()) {
		return false
	}
	for i := 0; i < t.NumElements(); i++ {
		diff := float32(math.Abs(float64(t.FlatAt(i) - other.FlatAt(i))))
		if diff > eps {
			return false
		}
	}
	return true
}
func (t Tensor) LogSoftmax() Tensor {
	return t.LogSoftmaxDim(0)
}
func (t Tensor) LogSoftmaxDim(axis int) Tensor {
	dims := t.Shape().Values()
	if len(dims) != 2 || axis != 1 {
		return t.LogSoftmax()
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
			sum += float32(math.Exp(float64(v - max)))
		}
		logSum := float32(math.Log(float64(sum)))
		for c := 0; c < cols; c++ {
			value := t.At(r, c)
			out.Set(value-max-logSum, r, c)
		}
	}
	return out
}
func (t Tensor) Log() Tensor {
	out := New(t.Shape())
	for i := 0; i < t.NumElements(); i++ {
		idx := t.memoryIndex(i)
		out.memory.Set(idx, float32(math.Log(float64(t.memory.At(idx)))))
	}
	return out
}
func (t Tensor) MatMul(other Tensor) (Tensor, bool) {
	a := t.shape.Values()
	b := other.shape.Values()
	if len(a) != 2 || len(b) != 2 {
		return Tensor{}, false
	}
	if a[1] != b[0] {
		return Tensor{}, false
	}
	out := New(shape.New(a[0], b[1]))
	for i := 0; i < a[0]; i++ {
		for j := 0; j < b[1]; j++ {
			var sum float32
			for k := 0; k < a[1]; k++ {
				av := t.At(i, k)
				bv := other.At(k, j)
				sum += av * bv
			}
			out.Set(sum, i, j)
		}
	}
	return out, true
}
func (t Tensor) View(newShape shape.Shape, offset int) (Tensor, bool) {
	if offset < 0 {
		return Tensor{}, false
	}
	if offset+newShape.NumElements() >
		t.memory.Len()-t.offset {
		return Tensor{}, false
	}
	return Tensor{
		shape:  newShape,
		stride: stride.Compute(newShape, t.layout),
		memory: t.memory,
		offset: t.offset + offset,
		device: t.device,
		layout: t.layout,
	}, true
}
func (t Tensor) Slice(start int, end int) (Tensor, bool) {
	dims := t.shape.Values()
	if len(dims) == 0 {
		return Tensor{}, false
	}
	if start < 0 || end > dims[0] || start >= end {
		return Tensor{}, false
	}
	newDims := make([]int, len(dims))
	copy(newDims, dims)
	newDims[0] = end - start
	offset := start * t.stride.At(0)
	return t.View(shape.New(newDims...), offset)
}
func (t Tensor) Clone() Tensor {
	return Tensor{shape: t.shape, stride: t.stride, memory: t.memory.Clone(), device: t.device, layout: t.layout}
}
func (t Tensor) Tanh() Tensor {
	out := t.Clone()
	for i := 0; i < out.NumElements(); i++ {
		idx := out.memoryIndex(i)
		v := out.memory.At(idx)
		out.memory.Set(idx, float32(math.Tanh(float64(v))))
	}
	return out
}
func (t Tensor) TanhBackward(grad Tensor) Tensor {
	out := New(t.Shape())
	for i := 0; i < t.NumElements(); i++ {
		idx := t.memoryIndex(i)
		v := t.memory.At(idx)
		g := grad.memory.At(grad.memoryIndex(i))
		out.memory.Set(out.memoryIndex(i), g*(1-v*v))
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
func (t Tensor) SoftmaxBackward(grad Tensor, axis int) Tensor {
	if axis != 1 {
		panic("SoftmaxBackward currently supports axis=1")
	}
	shape := t.Shape().Values()
	batch := shape[0]
	classes := shape[1]
	out := New(t.Shape())
	for n := range batch {
		dot := float32(0)
		for c := range classes {
			dot += grad.At(n, c) * t.At(n, c)
		}
		for c := range classes {
			y := t.At(n, c)
			dx := y * (grad.At(n, c) - dot)
			out.Set(dx, n, c)
		}
	}
	return out
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
func (t Tensor) LogSoftmaxBackward(grad Tensor, axis int) Tensor {
	if axis != 1 {
		panic("LogSoftmaxBackward currently supports axis=1")
	}
	shape := t.Shape().Values()
	batch := shape[0]
	classes := shape[1]
	out := New(t.Shape())
	for n := range batch {
		sumGrad := float32(0)
		for c := range classes {
			sumGrad += grad.At(n, c)
		}
		for c := range classes {
			softmax := float32(math.Exp(float64(t.At(n, c))))
			dx := grad.At(n, c) - softmax*sumGrad
			out.Set(dx, n, c)
		}
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
func (t Tensor) FlattenFrom(dim int) (Tensor, bool) {
	dims := t.Shape().Values()
	if dim < 0 || dim >= len(dims) {
		return Tensor{}, false
	}
	newDims := make([]int, 0, dim+1)
	newDims = append(newDims, dims[:dim]...)
	size := 1
	for _, d := range dims[dim:] {
		size *= d
	}
	newDims = append(newDims, size)
	return t.Reshape(shape.New(newDims...))
}

//	func (t Tensor) Flatten() (Tensor, bool) {
//		return t.FlattenFrom(1)
//	}
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
func (t Tensor) ReduceSumAxis(axis int) Tensor {
	dims := t.Shape().Values()
	if len(dims) != 2 {
		panic("ReduceSumAxis currently supports only 2D tensors")
	}
	rows := dims[0]
	cols := dims[1]
	switch axis {
	case 0:
		out := New(shape.New(cols))
		for c := range cols {
			sum := float32(0)
			for r := range rows {
				sum += t.At(r, c)
			}
			out.Set(sum, c)
		}
		return out
	case 1:
		out := New(shape.New(rows))
		for r := range rows {
			sum := float32(0)
			for c := range cols {
				sum += t.At(r, c)
			}
			out.Set(sum, r)
		}
		return out
	default:
		panic("invalid axis")
	}
}
func (t Tensor) LeakyReLU(alpha float32) Tensor {
	out := t.Clone()
	for i := 0; i < out.Len(); i++ {
		v := out.FlatAt(i)
		if v < 0 {
			v *= alpha
		}
		out.FlatSet(i, v)
	}
	return out
}
func (t Tensor) LeakyReLUBackward(grad Tensor, alpha float32) Tensor {
	out := New(t.Shape())
	for i := 0; i < t.Len(); i++ {
		g := grad.FlatAt(i)
		x := t.FlatAt(i)
		if x < 0 {
			g *= alpha
		}
		out.FlatSet(i, g)
	}
	return out
}
func (t Tensor) ELU(alpha float32) Tensor {
	out := New(t.Shape())
	for i := 0; i < t.Len(); i++ {
		x := t.FlatAt(i)
		if x >= 0 {
			out.FlatSet(i, x)
		} else {
			out.FlatSet(i, alpha*(float32(math.Exp(float64(x)))-1))
		}
	}
	return out
}
func (t Tensor) ELUBackward(grad Tensor, alpha float32) Tensor {
	out := grad.Clone()
	for i := 0; i < out.Len(); i++ {
		x := t.FlatAt(i)
		g := grad.FlatAt(i)
		if x > 0 {
			out.FlatSet(i, g)
		} else {
			out.FlatSet(i, g*alpha*float32(math.Exp(float64(x))))
		}
	}
	return out
}
func (t Tensor) ELUBackwardFromOutput(grad Tensor, alpha float32) Tensor {
	out := grad.Clone()
	for i := 0; i < out.Len(); i++ {
		y := t.FlatAt(i)
		g := grad.FlatAt(i)
		if y > 0 {
			out.FlatSet(i, g)
		} else {
			out.FlatSet(i, g*(y+alpha))
		}
	}
	return out
}
func (t Tensor) GELU() Tensor {
	out := New(t.Shape())
	for i := 0; i < t.Len(); i++ {
		x := float64(t.FlatAt(i))
		y := 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
		out.FlatSet(i, float32(y))
	}
	return out
}
func (t Tensor) GELUBackward(grad Tensor) Tensor {
	out := grad.Clone()
	const c = 0.044715
	const sqrt2pi = 0.7978845608
	for i := 0; i < out.Len(); i++ {
		x := float64(t.FlatAt(i))
		u := sqrt2pi * (x + c*x*x*x)
		th := math.Tanh(u)
		du := sqrt2pi * (1 + 3*c*x*x)
		dx := 0.5*(1+th) + 0.5*x*(1-th*th)*du
		out.FlatSet(i, grad.FlatAt(i)*float32(dx))
	}
	return out
}
func (t Tensor) Softplus() Tensor {
	out := New(t.Shape())
	for i := 0; i < t.Len(); i++ {
		x := float64(t.FlatAt(i))
		out.FlatSet(i, float32(math.Log(1+math.Exp(x))))
	}
	return out
}
func (t Tensor) SoftplusBackward(grad Tensor) Tensor {
	out := grad.Clone()
	for i := 0; i < out.Len(); i++ {
		x := t.FlatAt(i)
		// 		sig := float32(1.0 / (1.0 + math.Exp(-float64(x))))
		sig := float32(1.0 / (1.0 + math.Exp(float64(-x))))
		out.FlatSet(i, grad.FlatAt(i)*sig)
	}
	return out
}
func (t Tensor) Swish() Tensor {
	out := New(t.Shape())
	for i := 0; i < t.Len(); i++ {
		x := t.FlatAt(i)
		s := 1 / (1 + float32(math.Exp(-float64(x))))
		out.FlatSet(i, x*s)
	}
	return out
}
func (t Tensor) SwishBackward(grad Tensor) Tensor {
	out := grad.Clone()
	for i := 0; i < out.Len(); i++ {
		x := t.FlatAt(i)
		// sig := float32(1 / (1 + math.Exp(-float64(x))))
		sig := float32(1.0 / (1.0 + math.Exp(float64(-x))))
		d := sig + x*sig*(1-sig)
		out.FlatSet(i, grad.FlatAt(i)*d)
	}
	return out
}
func (t Tensor) Mish() Tensor {
	out := New(t.Shape())
	for i := 0; i < t.Len(); i++ {
		x := float64(t.FlatAt(i))
		sp := math.Log(1 + math.Exp(x))
		out.FlatSet(i, float32(x*math.Tanh(sp)))
	}
	return out
}
func (t Tensor) MishBackward(grad Tensor) Tensor {
	out := grad.Clone()
	for i := 0; i < out.Len(); i++ {
		x := t.FlatAt(i)
		sp := float32(math.Log1p(math.Exp(float64(x))))
		tanhSp := float32(math.Tanh(float64(sp)))
		sig := float32(1.0 / (1.0 + math.Exp(float64(-x))))
		dx := tanhSp + x*sig*(1-tanhSp*tanhSp)
		out.FlatSet(i, grad.FlatAt(i)*dx)
	}
	return out
}
func (t Tensor) HardSigmoid() Tensor {
	out := New(t.Shape())
	for i := 0; i < t.Len(); i++ {
		x := t.FlatAt(i)
		v := x/6 + 0.5
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		out.FlatSet(i, v)
	}
	return out
}
func (t Tensor) HardSigmoidBackward(grad Tensor) Tensor {
	out := New(t.Shape())
	for i := 0; i < out.Len(); i++ {
		x := t.FlatAt(i)
		if x <= -3 || x >= 3 {
			out.FlatSet(i, 0)
		} else {
			out.FlatSet(i, grad.FlatAt(i)/6)
		}
	}
	return out
}
func (t Tensor) HardSwish() Tensor {
	out := New(t.Shape())
	for i := 0; i < out.Len(); i++ {
		x := t.FlatAt(i)
		var y float32
		switch {
		case x <= -3:
			y = 0
		case x >= 3:
			y = x
		default:
			y = x * (x + 3) / 6
		}
		out.FlatSet(i, y)
	}
	return out
}
func (t Tensor) HardSwishBackward(grad Tensor) Tensor {
	out := New(t.Shape())
	for i := 0; i < out.Len(); i++ {
		x := t.FlatAt(i)
		var dx float32
		switch {
		case x <= -3:
			dx = 0
		case x >= 3:
			dx = 1
		default:
			dx = (2*x + 3) / 6
		}
		out.FlatSet(i, grad.FlatAt(i)*dx)
	}
	return out
}
func (t Tensor) BatchNorm(gamma Tensor, beta Tensor, eps float32) Tensor {
	dims := t.Shape().Values()
	if len(dims) != 2 {
		panic("BatchNorm supports only 2D tensors")
	}
	batch := dims[0]
	features := dims[1]
	out := New(t.Shape())
	for f := 0; f < features; f++ {
		mean := float32(0)
		for b := 0; b < batch; b++ {
			mean += t.At(b, f)
		}
		mean /= float32(batch)
		variance := float32(0)
		for b := 0; b < batch; b++ {
			d := t.At(b, f) - mean
			variance += d * d
		}
		variance /= float32(batch)
		invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))
		for b := 0; b < batch; b++ {
			xhat := (t.At(b, f) - mean) * invStd
			y := xhat*gamma.FlatAt(f) + beta.FlatAt(f)
			out.Set(y, b, f)
		}
	}
	return out
}
func (t Tensor) BatchNormBackward(grad Tensor) Tensor {
	return grad.Clone()
}
func (t Tensor) LayerNorm(gamma Tensor, beta Tensor, eps float32) (Tensor, Tensor, Tensor) {
	out := New(t.Shape())
	mean := Scalar(0)
	variance := Scalar(0)
	var m float32
	for i := 0; i < t.Len(); i++ {
		m += t.FlatAt(i)
	}
	m /= float32(t.Len())
	mean.FlatSet(0, m)
	var v float32
	for i := 0; i < t.Len(); i++ {
		d := t.FlatAt(i) - m
		v += d * d
	}
	v /= float32(t.Len())
	variance.FlatSet(0, v)
	denom := float32(math.Sqrt(float64(v + eps)))
	for i := 0; i < t.Len(); i++ {
		norm := (t.FlatAt(i) - m) / denom
		out.FlatSet(i, norm*gamma.FlatAt(i)+beta.FlatAt(i))
	}
	return out, mean, variance
}
func (t Tensor) LayerNormBackward(grad Tensor, gamma Tensor, mean Tensor, variance Tensor, eps float32) (Tensor, Tensor, Tensor) {
	dx := New(t.Shape())
	dgamma := New(gamma.Shape())
	dbeta := New(gamma.Shape())
	m := mean.FlatAt(0)
	v := variance.FlatAt(0)
	std := float32(math.Sqrt(float64(v + eps)))
	n := float32(t.Len())
	sumDy := float32(0)
	sumDyNorm := float32(0)
	for i := 0; i < t.Len(); i++ {
		norm := (t.FlatAt(i) - m) / std
		dy := grad.FlatAt(i)
		sumDy += dy
		sumDyNorm += dy * norm
		dbeta.FlatSet(i, dy)
		dgamma.FlatSet(i, dy*norm)
	}
	for i := 0; i < t.Len(); i++ {
		norm := (t.FlatAt(i) - m) / std
		dy := grad.FlatAt(i)
		g := gamma.FlatAt(i)
		value := (g / std) * (n*dy - sumDy - norm*sumDyNorm) / n
		dx.FlatSet(i, value)
	}
	return dx, dgamma, dbeta
}
func (t Tensor) L1Loss(target Tensor) Tensor {
	diff := t.Sub(target)
	out := New(diff.Shape())
	for i := 0; i < diff.Len(); i++ {
		v := diff.FlatAt(i)
		if v < 0 {
			v = -v
		}
		out.FlatSet(i, v)
	}
	return out
}
func (t Tensor) L1LossBackward(target Tensor, grad Tensor) (Tensor, Tensor) {
	dPred := New(t.Shape())
	dTarget := New(target.Shape())
	for i := 0; i < t.Len(); i++ {
		p := t.FlatAt(i)
		y := target.FlatAt(i)
		g := grad.FlatAt(i)
		switch {
		case p > y:
			dPred.FlatSet(i, g)
			dTarget.FlatSet(i, -g)
		case p < y:
			dPred.FlatSet(i, -g)
			dTarget.FlatSet(i, g)

		default:
			dPred.FlatSet(i, 0)
			dTarget.FlatSet(i, 0)
		}
	}
	return dPred, dTarget
}
func (t Tensor) SmoothL1Loss(target Tensor, beta float32) Tensor {

	if beta <= 0 {
		return t.L1Loss(target)
	}

	out := New(t.Shape())

	for i := 0; i < t.Len(); i++ {

		diff := t.FlatAt(i) - target.FlatAt(i)

		if diff < 0 {
			diff = -diff
		}

		if diff < beta {
			out.FlatSet(i, 0.5*diff*diff/beta)
		} else {
			out.FlatSet(i, diff-0.5*beta)
		}
	}

	return out
}
func (t Tensor) SmoothL1LossBackward(
	target Tensor,
	grad Tensor,
	beta float32,
) (Tensor, Tensor) {

	dPred := New(t.Shape())
	dTarget := New(target.Shape())

	for i := 0; i < t.Len(); i++ {

		diff := t.FlatAt(i) - target.FlatAt(i)

		var g float32

		switch {

		case diff > beta:
			g = 1

		case diff < -beta:
			g = -1

		default:
			g = diff / beta
		}

		g *= grad.FlatAt(i)

		dPred.FlatSet(i, g)
		dTarget.FlatSet(i, -g)
	}

	return dPred, dTarget
}
func (t Tensor) HuberLoss(target Tensor, delta float32) Tensor {

	out := New(t.Shape())

	for i := 0; i < t.Len(); i++ {

		diff := t.FlatAt(i) - target.FlatAt(i)

		abs := diff
		if abs < 0 {
			abs = -abs
		}

		if abs <= delta {
			out.FlatSet(i, 0.5*diff*diff)
		} else {
			out.FlatSet(i, delta*(abs-0.5*delta))
		}
	}

	return out
}
func (t Tensor) HuberLossBackward(
	target Tensor,
	grad Tensor,
	delta float32,
) (Tensor, Tensor) {

	dPred := New(t.Shape())
	dTarget := New(target.Shape())

	for i := 0; i < t.Len(); i++ {

		diff := t.FlatAt(i) - target.FlatAt(i)

		var g float32

		switch {

		case diff > delta:
			g = delta

		case diff < -delta:
			g = -delta

		default:
			g = diff
		}

		g *= grad.FlatAt(i)

		dPred.FlatSet(i, g)
		dTarget.FlatSet(i, -g)
	}

	return dPred, dTarget
}
func (t Tensor) BCELoss(target Tensor) Tensor {

	out := New(t.Shape())

	const eps = 1e-7

	for i := 0; i < t.Len(); i++ {

		p := t.FlatAt(i)
		y := target.FlatAt(i)

		if p < eps {
			p = eps
		}

		if p > 1-eps {
			p = 1 - eps
		}

		loss :=
			-y*float32(math.Log(float64(p))) -
				(1-y)*float32(math.Log(float64(1-p)))

		out.FlatSet(i, loss)
	}

	return out
}
func (t Tensor) BCELossBackward(
	target Tensor,
	grad Tensor,
) (Tensor, Tensor) {

	dPred := New(t.Shape())
	dTarget := New(target.Shape())

	const eps = 1e-7

	for i := 0; i < t.Len(); i++ {

		p := t.FlatAt(i)
		y := target.FlatAt(i)

		if p < eps {
			p = eps
		}

		if p > 1-eps {
			p = 1 - eps
		}

		g := grad.FlatAt(i)

		dp :=
			(-y/p +
				(1-y)/(1-p)) * g

		dPred.FlatSet(i, dp)

		dTarget.FlatSet(i, 0)
	}

	return dPred, dTarget
}
func (t Tensor) BCEWithLogitsLoss(target Tensor) Tensor {

	out := New(t.Shape())

	for i := 0; i < t.Len(); i++ {

		x := t.FlatAt(i)
		y := target.FlatAt(i)

		loss :=
			maxf(x, 0) -
				x*y +
				float32(math.Log1p(math.Exp(float64(-absf(x)))))

		out.FlatSet(i, loss)
	}

	return out
}
func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func absf(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
func (t Tensor) BCEWithLogitsBackward(
	target Tensor,
	grad Tensor,
) (Tensor, Tensor) {

	dPred := New(t.Shape())
	dTarget := New(target.Shape())

	for i := 0; i < t.Len(); i++ {

		x := t.FlatAt(i)

		sig :=
			float32(
				1.0 /
					(1.0 + math.Exp(-float64(x))),
			)

		g :=
			(sig - target.FlatAt(i)) *
				grad.FlatAt(i)

		dPred.FlatSet(i, g)

		dTarget.FlatSet(i, 0)
	}

	return dPred, dTarget
}
func (t Tensor) Conv2D(
	weight Tensor,
	bias Tensor,
	strideH,
	strideW,
	padH,
	padW int,
) Tensor {

	xShape := t.Shape().Values()
	wShape := weight.Shape().Values()

	batch := xShape[0]
	inC := xShape[1]
	height := xShape[2]
	width := xShape[3]

	outC := wShape[0]
	kernelH := wShape[2]
	kernelW := wShape[3]

	outH := (height+2*padH-kernelH)/strideH + 1
	outW := (width+2*padW-kernelW)/strideW + 1

	out := New(shape.New(batch, outC, outH, outW))

	for n := 0; n < batch; n++ {
		for oc := 0; oc < outC; oc++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {

					sum := bias.At(oc)

					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kernelH; kh++ {
							for kw := 0; kw < kernelW; kw++ {

								ih := oh*strideH + kh - padH
								iw := ow*strideW + kw - padW

								if ih < 0 || ih >= height {
									continue
								}

								if iw < 0 || iw >= width {
									continue
								}

								sum +=
									t.At(n, ic, ih, iw) *
										weight.At(oc, ic, kh, kw)

							}
						}
					}

					out.Set(sum, n, oc, oh, ow)
				}
			}
		}
	}

	return out
}
func (t Tensor) Conv2DBackward(
	grad Tensor,
	weight Tensor,
	strideH,
	strideW,
	padH,
	padW int,
) (Tensor, Tensor, Tensor) {

	xShape := t.Shape().Values()
	wShape := weight.Shape().Values()

	batch := xShape[0]
	inC := xShape[1]
	height := xShape[2]
	width := xShape[3]

	outC := wShape[0]
	kernelH := wShape[2]
	kernelW := wShape[3]

	outH := grad.Shape().Values()[2]
	outW := grad.Shape().Values()[3]

	dx := New(t.Shape())
	dw := New(weight.Shape())
	db := New(shape.New(outC))

	for n := 0; n < batch; n++ {

		for oc := 0; oc < outC; oc++ {

			for oh := 0; oh < outH; oh++ {

				for ow := 0; ow < outW; ow++ {

					g := grad.At(n, oc, oh, ow)

					db.Set(
						db.At(oc)+g,
						oc,
					)

					for ic := 0; ic < inC; ic++ {

						for kh := 0; kh < kernelH; kh++ {

							for kw := 0; kw < kernelW; kw++ {

								ih := oh*strideH + kh - padH
								iw := ow*strideW + kw - padW

								if ih < 0 || ih >= height {
									continue
								}

								if iw < 0 || iw >= width {
									continue
								}

								dw.Set(
									dw.At(oc, ic, kh, kw)+
										g*t.At(n, ic, ih, iw),
									oc,
									ic,
									kh,
									kw,
								)

								dx.Set(
									dx.At(n, ic, ih, iw)+
										g*weight.At(oc, ic, kh, kw),
									n,
									ic,
									ih,
									iw,
								)

							}
						}
					}
				}
			}
		}
	}

	return dx, dw, db
}
func (t Tensor) MaxPool2D(
	kernelH, kernelW,
	strideH, strideW int,
) (Tensor, []int) {

	dims := t.Shape().Values()

	batch := dims[0]
	channels := dims[1]
	height := dims[2]
	width := dims[3]

	outH := (height-kernelH)/strideH + 1
	outW := (width-kernelW)/strideW + 1

	out := New(shape.New(batch, channels, outH, outW))

	argmax := make([]int, batch*channels*outH*outW)

	index := 0

	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			for oh := 0; oh < outH; oh++ {

				for ow := 0; ow < outW; ow++ {

					maxValue := float32(math.Inf(-1))
					maxIndex := 0

					for kh := 0; kh < kernelH; kh++ {

						for kw := 0; kw < kernelW; kw++ {

							ih := oh*strideH + kh
							iw := ow*strideW + kw

							v := t.At(n, c, ih, iw)

							if v > maxValue {

								maxValue = v

								maxIndex =
									((n*channels+c)*height+ih)*width + iw
							}
						}
					}

					out.Set(maxValue, n, c, oh, ow)

					argmax[index] = maxIndex

					index++
				}
			}
		}
	}

	return out, argmax
}
func (t Tensor) MaxPool2DBackward(
	grad Tensor,
	argmax []int,
) Tensor {

	dx := New(t.Shape())

	dims := grad.Shape().Values()

	batch := dims[0]
	channels := dims[1]
	outH := dims[2]
	outW := dims[3]

	index := 0

	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			for h := 0; h < outH; h++ {

				for w := 0; w < outW; w++ {

					i := argmax[index]

					dx.FlatSet(
						i,
						dx.FlatAt(i)+grad.At(n, c, h, w),
					)

					index++
				}
			}
		}
	}

	return dx
}
func (t Tensor) AvgPool2D(
	kernelH, kernelW,
	strideH, strideW int,
) Tensor {

	dims := t.Shape().Values()

	batch := dims[0]
	channels := dims[1]
	height := dims[2]
	width := dims[3]

	outH := (height-kernelH)/strideH + 1
	outW := (width-kernelW)/strideW + 1

	out := New(shape.New(batch, channels, outH, outW))

	scale := float32(1.0 / float32(kernelH*kernelW))

	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			for oh := 0; oh < outH; oh++ {

				for ow := 0; ow < outW; ow++ {

					sum := float32(0)

					for kh := 0; kh < kernelH; kh++ {

						for kw := 0; kw < kernelW; kw++ {

							ih := oh*strideH + kh
							iw := ow*strideW + kw

							sum += t.At(n, c, ih, iw)
						}
					}

					out.Set(sum*scale, n, c, oh, ow)
				}
			}
		}
	}

	return out
}
func (t Tensor) AvgPool2DBackward(
	grad Tensor,
	kernelH, kernelW,
	strideH, strideW int,
) Tensor {

	dx := New(t.Shape())

	dims := t.Shape().Values()

	batch := dims[0]
	channels := dims[1]

	outH := grad.Shape().Values()[2]
	outW := grad.Shape().Values()[3]

	scale := float32(1.0 / float32(kernelH*kernelW))

	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			for oh := 0; oh < outH; oh++ {

				for ow := 0; ow < outW; ow++ {

					g := grad.At(n, c, oh, ow) * scale

					for kh := 0; kh < kernelH; kh++ {

						for kw := 0; kw < kernelW; kw++ {

							ih := oh*strideH + kh
							iw := ow*strideW + kw

							dx.Set(
								dx.At(n, c, ih, iw)+g,
								n, c, ih, iw,
							)
						}
					}
				}
			}
		}
	}

	return dx
}
func (t Tensor) AdaptiveAvgPool2D(
	outH, outW int,
) Tensor {

	dims := t.Shape().Values()

	batch := dims[0]
	channels := dims[1]
	inH := dims[2]
	inW := dims[3]

	out := New(shape.New(batch, channels, outH, outW))

	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			for oh := 0; oh < outH; oh++ {

				hStart := oh * inH / outH
				hEnd := (oh + 1) * inH / outH

				for ow := 0; ow < outW; ow++ {

					wStart := ow * inW / outW
					wEnd := (ow + 1) * inW / outW

					sum := float32(0)

					count := (hEnd - hStart) * (wEnd - wStart)

					for h := hStart; h < hEnd; h++ {

						for w := wStart; w < wEnd; w++ {

							sum += t.At(n, c, h, w)

						}
					}

					out.Set(
						sum/float32(count),
						n, c, oh, ow,
					)
				}
			}
		}
	}

	return out
}
func (t Tensor) AdaptiveAvgPool2DBackward(
	grad Tensor,
	outH,
	outW int,
) Tensor {

	dims := t.Shape().Values()

	batch := dims[0]
	channels := dims[1]
	inH := dims[2]
	inW := dims[3]

	dx := New(t.Shape())

	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			for oh := 0; oh < outH; oh++ {

				hStart := oh * inH / outH
				hEnd := (oh + 1) * inH / outH

				for ow := 0; ow < outW; ow++ {

					wStart := ow * inW / outW
					wEnd := (ow + 1) * inW / outW

					g := grad.At(n, c, oh, ow)

					scale := g / float32(
						(hEnd-hStart)*(wEnd-wStart),
					)

					for h := hStart; h < hEnd; h++ {

						for w := wStart; w < wEnd; w++ {

							dx.Set(
								dx.At(n, c, h, w)+scale,
								n, c, h, w,
							)

						}
					}
				}
			}
		}
	}

	return dx
}
func (t Tensor) AdaptiveMaxPool2D(
	outH, outW int,
) (Tensor, []int) {

	dims := t.Shape().Values()

	if len(dims) != 4 {
		panic("AdaptiveMaxPool2D expects NCHW tensor")
	}

	batch := dims[0]
	channels := dims[1]
	inH := dims[2]
	inW := dims[3]

	out := New(shape.New(batch, channels, outH, outW))

	argmax := make([]int, batch*channels*outH*outW)

	index := 0

	for n := 0; n < batch; n++ {
		for c := 0; c < channels; c++ {

			for oh := 0; oh < outH; oh++ {

				hStart := (oh * inH) / outH
				hEnd := ((oh + 1) * inH) / outH

				for ow := 0; ow < outW; ow++ {

					wStart := (ow * inW) / outW
					wEnd := ((ow + 1) * inW) / outW

					maxVal := float32(math.Inf(-1))
					maxIdx := 0

					for h := hStart; h < hEnd; h++ {
						for w := wStart; w < wEnd; w++ {

							v := t.At(n, c, h, w)

							if v > maxVal {
								maxVal = v
								maxIdx = ((n*channels+c)*inH+h)*inW + w
							}
						}
					}

					out.Set(maxVal, n, c, oh, ow)
					argmax[index] = maxIdx
					index++
				}
			}
		}
	}

	return out, argmax
}
func (t Tensor) AdaptiveMaxPool2DBackward(
	grad Tensor,
	argmax []int,
) Tensor {

	dx := New(t.Shape())

	dims := grad.Shape().Values()

	batch := dims[0]
	channels := dims[1]
	outH := dims[2]
	outW := dims[3]

	index := 0

	for n := 0; n < batch; n++ {
		for c := 0; c < channels; c++ {
			for h := 0; h < outH; h++ {
				for w := 0; w < outW; w++ {

					inputIndex := argmax[index]

					old := dx.FlatAt(inputIndex)

					dx.FlatSet(
						inputIndex,
						old+grad.At(n, c, h, w),
					)

					index++
				}
			}
		}
	}

	return dx
}
func (t Tensor) ConvTranspose2D(
	weight Tensor,
	bias Tensor,
	strideH,
	strideW,
	paddingH,
	paddingW int,
) Tensor {

	inShape := t.Shape().Values()
	wShape := weight.Shape().Values()

	batch := inShape[0]
	inC := inShape[1]
	inH := inShape[2]
	inW := inShape[3]

	outC := wShape[1]
	kernelH := wShape[2]
	kernelW := wShape[3]

	outH := (inH-1)*strideH - 2*paddingH + kernelH
	outW := (inW-1)*strideW - 2*paddingW + kernelW

	out := New(shape.New(
		batch,
		outC,
		outH,
		outW,
	))

	for n := 0; n < batch; n++ {

		for ic := 0; ic < inC; ic++ {

			for ih := 0; ih < inH; ih++ {

				for iw := 0; iw < inW; iw++ {

					input := t.At(
						n,
						ic,
						ih,
						iw,
					)

					for oc := 0; oc < outC; oc++ {

						for kh := 0; kh < kernelH; kh++ {

							for kw := 0; kw < kernelW; kw++ {

								oh :=
									ih*strideH +
										kh -
										paddingH

								ow :=
									iw*strideW +
										kw -
										paddingW

								if oh < 0 ||
									oh >= outH ||
									ow < 0 ||
									ow >= outW {

									continue
								}

								old := out.At(
									n,
									oc,
									oh,
									ow,
								)

								old +=
									input *
										weight.At(
											ic,
											oc,
											kh,
											kw,
										)

								out.Set(
									old,
									n,
									oc,
									oh,
									ow,
								)

							}
						}
					}
				}
			}
		}
	}

	for n := 0; n < batch; n++ {

		for oc := 0; oc < outC; oc++ {

			b := bias.At(oc)

			for h := 0; h < outH; h++ {

				for w := 0; w < outW; w++ {

					out.Set(
						out.At(
							n,
							oc,
							h,
							w,
						)+b,
						n,
						oc,
						h,
						w,
					)

				}
			}
		}
	}

	return out
}
func (t Tensor) ConvTranspose2DBackward(
	grad Tensor,
	weight Tensor,
	strideH,
	strideW,
	paddingH,
	paddingW int,
) (Tensor, Tensor, Tensor) {

	inShape := t.Shape().Values()
	wShape := weight.Shape().Values()
	gShape := grad.Shape().Values()

	batch := inShape[0]
	inC := inShape[1]
	inH := inShape[2]
	inW := inShape[3]

	outC := wShape[1]
	kernelH := wShape[2]
	kernelW := wShape[3]

	outH := gShape[2]
	outW := gShape[3]

	dx := New(t.Shape())
	dw := New(weight.Shape())
	db := New(shape.New(outC))

	// db
	for n := 0; n < batch; n++ {
		for oc := 0; oc < outC; oc++ {
			sum := db.At(oc)

			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					sum += grad.At(n, oc, oh, ow)
				}
			}

			db.Set(sum, oc)
		}
	}

	// dx + dw
	for n := 0; n < batch; n++ {

		for ic := 0; ic < inC; ic++ {

			for ih := 0; ih < inH; ih++ {

				for iw := 0; iw < inW; iw++ {

					for oc := 0; oc < outC; oc++ {

						for kh := 0; kh < kernelH; kh++ {

							for kw := 0; kw < kernelW; kw++ {

								oh := ih*strideH + kh - paddingH
								ow := iw*strideW + kw - paddingW

								if oh < 0 ||
									oh >= outH ||
									ow < 0 ||
									ow >= outW {
									continue
								}

								g := grad.At(
									n,
									oc,
									oh,
									ow,
								)

								// dWeight
								dw.Set(
									dw.At(ic, oc, kh, kw)+
										t.At(n, ic, ih, iw)*g,
									ic,
									oc,
									kh,
									kw,
								)

								// dInput
								dx.Set(
									dx.At(n, ic, ih, iw)+
										weight.At(ic, oc, kh, kw)*g,
									n,
									ic,
									ih,
									iw,
								)
							}
						}
					}
				}
			}
		}
	}

	return dx, dw, db
}
func reflectIndex(index, size int) int {
	if size <= 1 {
		return 0
	}
	for index < 0 || index >= size {
		if index < 0 {
			index = -index
		}
		if index >= size {
			index = 2*size - index - 2
		}
	}
	return index
}
func clamp(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
func (t Tensor) ReflectionPad2D(
	left,
	right,
	top,
	bottom int,
) Tensor {

	d := t.Shape().Values()

	if len(d) != 4 {
		panic("ReflectionPad2D expects NCHW tensor")
	}

	n := d[0]
	c := d[1]
	h := d[2]
	w := d[3]

	out := New(shape.New(
		n,
		c,
		h+top+bottom,
		w+left+right,
	))
	for bn := 0; bn < n; bn++ {

		for ch := 0; ch < c; ch++ {

			for y := 0; y < h+top+bottom; y++ {

				for x := 0; x < w+left+right; x++ {

					iy := reflectIndex(y-top, h)
					ix := reflectIndex(x-left, w)

					out.Set(
						t.At(bn, ch, iy, ix),
						bn,
						ch,
						y,
						x,
					)
				}
			}
		}
	}

	return out
}
func (t Tensor) ReflectionPad2DBackward(
	grad Tensor,
	left,
	right,
	top,
	bottom int,
) Tensor {

	inShape := t.Shape().Values()

	n := inShape[0]
	c := inShape[1]
	h := inShape[2]
	w := inShape[3]

	dx := New(t.Shape())

	reflectIndex := func(i, size int) int {

		if i < 0 {
			return -i
		}

		if i >= size {
			return 2*size - i - 2
		}

		return i
	}

	for bn := 0; bn < n; bn++ {

		for ch := 0; ch < c; ch++ {

			for y := 0; y < h+top+bottom; y++ {

				for x := 0; x < w+left+right; x++ {

					iy := reflectIndex(y-top, h)
					ix := reflectIndex(x-left, w)

					dx.Set(
						dx.At(bn, ch, iy, ix)+
							grad.At(bn, ch, y, x),
						bn,
						ch,
						iy,
						ix,
					)
				}
			}
		}
	}

	return dx
}
func (t Tensor) ReplicationPad2D(
	left,
	right,
	top,
	bottom int,
) Tensor {

	d := t.Shape().Values()

	if len(d) != 4 {
		panic("ReplicationPad2D expects NCHW tensor")
	}

	n := d[0]
	c := d[1]
	h := d[2]
	w := d[3]

	out := New(shape.New(
		n,
		c,
		h+top+bottom,
		w+left+right,
	))
	for bn := 0; bn < n; bn++ {
		for ch := 0; ch < c; ch++ {

			for y := 0; y < h+top+bottom; y++ {

				for x := 0; x < w+left+right; x++ {

					iy := clamp(y-top, 0, h-1)
					ix := clamp(x-left, 0, w-1)

					out.Set(
						t.At(bn, ch, iy, ix),
						bn,
						ch,
						y,
						x,
					)
				}
			}
		}
	}

	return out
}
func (t Tensor) ReplicationPad2DBackward(
	grad Tensor,
	left,
	right,
	top,
	bottom int,
) Tensor {

	d := t.Shape().Values()

	n := d[0]
	c := d[1]
	h := d[2]
	w := d[3]

	dx := New(t.Shape())

	clamp := func(v, min, max int) int {
		if v < min {
			return min
		}
		if v > max {
			return max
		}
		return v
	}

	for bn := 0; bn < n; bn++ {

		for ch := 0; ch < c; ch++ {

			for y := 0; y < h+top+bottom; y++ {

				for x := 0; x < w+left+right; x++ {

					iy := clamp(y-top, 0, h-1)
					ix := clamp(x-left, 0, w-1)

					dx.Set(
						dx.At(bn, ch, iy, ix)+
							grad.At(bn, ch, y, x),
						bn,
						ch,
						iy,
						ix,
					)
				}
			}
		}
	}

	return dx
}
func (t Tensor) PixelShuffle(upscale int) Tensor {

	dims := t.Shape().Values()

	if len(dims) != 4 {
		panic("PixelShuffle expects NCHW tensor")
	}

	n := dims[0]
	cin := dims[1]
	h := dims[2]
	w := dims[3]

	if cin%(upscale*upscale) != 0 {
		panic("invalid channel count")
	}

	cout := cin / (upscale * upscale)

	out := New(shape.New(
		n,
		cout,
		h*upscale,
		w*upscale,
	))

	for bn := 0; bn < n; bn++ {

		for co := 0; co < cout; co++ {

			for y := 0; y < h; y++ {

				for x := 0; x < w; x++ {

					for ry := 0; ry < upscale; ry++ {

						for rx := 0; rx < upscale; rx++ {

							ci :=
								co*upscale*upscale +
									ry*upscale +
									rx

							out.Set(
								t.At(
									bn,
									ci,
									y,
									x,
								),
								bn,
								co,
								y*upscale+ry,
								x*upscale+rx,
							)
						}
					}
				}
			}
		}
	}

	return out
}
func (t Tensor) PixelShuffleBackward(
	grad Tensor,
	upscale int,
) Tensor {

	d := t.Shape().Values()

	n := d[0]
	cin := d[1]
	h := d[2]
	w := d[3]

	cout := cin / (upscale * upscale)

	dx := New(t.Shape())

	for bn := 0; bn < n; bn++ {

		for co := 0; co < cout; co++ {

			for y := 0; y < h; y++ {

				for x := 0; x < w; x++ {

					for ry := 0; ry < upscale; ry++ {

						for rx := 0; rx < upscale; rx++ {

							ci :=
								co*upscale*upscale +
									ry*upscale +
									rx

							dx.Set(
								grad.At(
									bn,
									co,
									y*upscale+ry,
									x*upscale+rx,
								),
								bn,
								ci,
								y,
								x,
							)
						}
					}
				}
			}
		}
	}

	return dx
}
func (t Tensor) PixelUnshuffle(downscale int) Tensor {

	dims := t.Shape().Values()

	if len(dims) != 4 {
		panic("PixelUnshuffle expects NCHW tensor")
	}

	n := dims[0]
	cin := dims[1]
	h := dims[2]
	w := dims[3]

	if h%downscale != 0 || w%downscale != 0 {
		panic("height and width must be divisible by downscale")
	}

	cout := cin * downscale * downscale
	outh := h / downscale
	outw := w / downscale

	out := New(shape.New(
		n,
		cout,
		outh,
		outw,
	))

	for bn := 0; bn < n; bn++ {

		for co := 0; co < cin; co++ {

			for oy := 0; oy < outh; oy++ {

				for ox := 0; ox < outw; ox++ {

					for ry := 0; ry < downscale; ry++ {

						for rx := 0; rx < downscale; rx++ {

							ci :=
								co*downscale*downscale +
									ry*downscale +
									rx

							out.Set(
								t.At(
									bn,
									co,
									oy*downscale+ry,
									ox*downscale+rx,
								),
								bn,
								ci,
								oy,
								ox,
							)
						}
					}
				}
			}
		}
	}

	return out
}
func (t Tensor) PixelUnshuffleBackward(
	grad Tensor,
	downscale int,
) Tensor {

	dims := t.Shape().Values()

	n := dims[0]
	cin := dims[1]
	h := dims[2]
	w := dims[3]

	dx := New(t.Shape())

	cout := cin * downscale * downscale

	_ = cout

	for bn := 0; bn < n; bn++ {

		for co := 0; co < cin; co++ {

			for oy := 0; oy < h/downscale; oy++ {

				for ox := 0; ox < w/downscale; ox++ {

					for ry := 0; ry < downscale; ry++ {

						for rx := 0; rx < downscale; rx++ {

							ci :=
								co*downscale*downscale +
									ry*downscale +
									rx

							dx.Set(
								grad.At(
									bn,
									ci,
									oy,
									ox,
								),
								bn,
								co,
								oy*downscale+ry,
								ox*downscale+rx,
							)
						}
					}
				}
			}
		}
	}

	return dx
}

// func indicesToLinear(indices []int, sh shape.Shape) int {
// 	dims := sh.Values()
// 	stride := 1
// 	index := 0
// 	for i := len(dims) - 1; i >= 0; i-- {
// 		index += indices[i] * stride
// 		stride *= dims[i]
// 	}
// 	return index
// }
// func reduceMeanBackward(

// 	grad tensor.Tensor,
// 	inputShape shape.Shape,
// 	axis int,

// ) tensor.Tensor {
// 	out := tensor.New(inputShape)
// 	axisSize := inputShape[axis]
// 	for position := range inputShape {
// 		out.Set(position, gard.value/float32(axisSize))
// 	}
// 	return out
// }

// func reduceMaxBackward(
//
//	input tensor.Tensor,
//	grad tensor.Tensor,
//	axis int,
//
//	) tensor.Tensor {
//	    out := tensor.New(input.Shape())
//	    for output := range index {
//	        maxValue := grad.position
//	        for axis positions {
//	            if input == maxValue {
//	                out += grad
//	            }
//	        }
//	    }
//	    return out
//	}
// func ReduceMeanAxis(x *autograd.Variable, axis int) *autograd.Variable {
// 	outTensor := reduceMean(x.Data(), axis)
// 	out := autograd.NewVariable(outTensor, x.RequiresGrad())
// 	if x.RequiresGrad() {
// 		out.SetBackward(func(grad tensor.Tensor) {
// 			gx := reduceMeanBackward(grad, x.Shape(), axis)
// 			x.AccumulateGrad(gx)
// 		})
// 	}
// 	return out
// }
