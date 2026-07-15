package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestSum(t *testing.T) {

	tensor := From(
		shape.New(3),
		[]float32{
			1, 2, 3,
		},
	)

	if tensor.Sum() != 6 {

		t.Fatal()

	}

}

func TestMean(t *testing.T) {

	tensor := From(
		shape.New(4),
		[]float32{
			2, 4, 6, 8,
		},
	)

	if tensor.Mean() != 5 {

		t.Fatal()

	}

}
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
func TestSqueeze(t *testing.T) {

	x := New(shape.New(1, 3, 1))

	y := x.Squeeze()

	if !y.Shape().Equal(shape.New(3)) {
		t.Fatal("wrong shape")
	}
}
func TestIterator(t *testing.T) {

	tensor := From(
		shape.New(3),
		[]float32{
			1, 2, 3,
		},
	)

	it := tensor.Iterator()

	sum := float32(0)

	for it.HasNext() {

		sum += it.Next()

	}

	if sum != 6 {

		t.Fatal()

	}

}
func TestAdd(t *testing.T) {

	a := From(
		shape.New(3),
		[]float32{1, 2, 3},
	)

	b := From(
		shape.New(3),
		[]float32{4, 5, 6},
	)

	c := a.Add(b)

	if c.At(0) != 5 {

		t.Fatal()

	}

	if c.At(2) != 9 {

		t.Fatal()

	}

}
func TestAddInplace(t *testing.T) {

	a := From(
		shape.New(3),
		[]float32{1, 2, 3},
	)

	b := From(
		shape.New(3),
		[]float32{4, 5, 6},
	)

	ok := a.AddInplace(b)

	if !ok {

		t.Fatal()

	}

	if a.At(0) != 5 {

		t.Fatal()

	}

}
func TestPad2D(t *testing.T) {

	x :=
		New(
			shape.New(2, 2),
		)

	x.Set(1, 0, 0)
	x.Set(2, 0, 1)
	x.Set(3, 1, 0)
	x.Set(4, 1, 1)

	out, ok :=
		x.Pad(
			1,
			1,
			1,
			1,
			0,
		)

	if !ok {
		t.Fatal("pad failed")
	}

	values :=
		out.Shape().Values()

	if values[0] != 4 ||
		values[1] != 4 {

		t.Fatal("wrong pad shape")

	}

	if out.At(1, 1) != 1 {
		t.Fatal("wrong copied value")
	}

	if out.At(2, 2) != 4 {
		t.Fatal("wrong copied center")
	}

	if out.At(0, 0) != 0 {
		t.Fatal("wrong padding value")
	}

}
func TestMatMul(t *testing.T) {

	a := From(
		shape.New(2, 3),
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
	)

	b := From(
		shape.New(3, 2),
		[]float32{
			7, 8,
			9, 10,
			11, 12,
		},
	)

	c, ok := a.MatMul(b)

	if !ok {

		t.Fatal()

	}

	if c.At(0, 0) != 58 {

		t.Fatal()

	}

	if c.At(1, 1) != 154 {

		t.Fatal()

	}

}
func TestTranspose(t *testing.T) {

	a := From(
		shape.New(2, 3),
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
	)

	b, ok := a.Transpose()

	if !ok {

		t.Fatal()

	}

	if b.At(0, 1) != 4 {

		t.Fatal()

	}

}
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
