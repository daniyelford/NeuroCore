package ndim

import "testing"

func TestClone(t *testing.T) {

	v := New(2, 3, 4)

	c := v.Clone()

	if !v.Equal(c) {
		t.Fatal()
	}

	n := c.With(0, 100)

	if v.At(0) == n.At(0) {
		t.Fatal("clone modified original")
	}

}
func TestPrepend(t *testing.T) {

	v := New(3, 4)

	n := v.Prepend(1, 2)

	if !n.Equal(New(1, 2, 3, 4)) {
		t.Fatal()
	}

}

func TestRank(t *testing.T) {

	v := New(3, 4, 5)

	if v.Rank() != 3 {
		t.Fatal()
	}

}

func TestValid(t *testing.T) {

	if !New(2, 3).Valid() {
		t.Fatal()
	}

	if New(2, 0).Valid() {
		t.Fatal()
	}

}
func TestGet(t *testing.T) {

	v := New(10, 20, 30)

	value, ok := v.Get(1)

	if !ok {
		t.Fatal()
	}

	if value != 20 {
		t.Fatal()
	}

	_, ok = v.Get(100)

	if ok {
		t.Fatal()
	}

}
func TestAppend(t *testing.T) {

	v := New(1, 2)

	n := v.Append(3, 4)

	if !n.Equal(New(1, 2, 3, 4)) {
		t.Fatal()
	}

	if !v.Equal(New(1, 2)) {
		t.Fatal("original vector modified")
	}

}

func TestProduct(t *testing.T) {

	v := New(2, 3, 4)

	if v.Product() != 24 {
		t.Fatal()
	}

}

func TestSum(t *testing.T) {

	v := New(2, 3, 4)

	if v.Sum() != 9 {
		t.Fatal()
	}

}

func TestMax(t *testing.T) {

	v := New(5, 9, 1)

	if v.Max() != 9 {
		t.Fatal()
	}

}

func TestMin(t *testing.T) {

	v := New(5, 9, 1)

	if v.Min() != 1 {
		t.Fatal()
	}

}

func TestContains(t *testing.T) {

	v := New(2, 4, 6)

	if !v.Contains(4) {
		t.Fatal()
	}

	if v.Contains(10) {
		t.Fatal()
	}

}

func TestIndexOf(t *testing.T) {

	v := New(5, 8, 12)

	if v.IndexOf(8) != 1 {
		t.Fatal()
	}

	if v.IndexOf(100) != -1 {
		t.Fatal()
	}

}
func TestWith(t *testing.T) {

	v := New(1, 2, 3)

	n := v.With(1, 100)

	if n.At(1) != 100 {
		t.Fatal()
	}

	if v.At(1) != 2 {
		t.Fatal("original vector modified")
	}

}

func TestRemove(t *testing.T) {

	v := New(1, 2, 3)

	n := v.Remove(1)

	if !n.Equal(New(1, 3)) {
		t.Fatal()
	}

}

func TestSlice(t *testing.T) {

	v := New(1, 2, 3, 4)

	n := v.Slice(1, 3)

	if !n.Equal(New(2, 3)) {
		t.Fatal()
	}

}

func TestReverse(t *testing.T) {

	v := New(1, 2, 3)

	n := v.Reverse()

	if !n.Equal(New(3, 2, 1)) {
		t.Fatal()
	}

	if !v.Equal(New(1, 2, 3)) {
		t.Fatal("original vector modified")
	}

}
func TestRange(t *testing.T) {

	v := New(2, 4, 6)

	sum := 0

	v.Range(func(i, value int) bool {

		sum += value

		return true

	})

	if sum != 12 {
		t.Fatal()
	}

}
func TestIsScalar(t *testing.T) {

	if !New(5).IsScalar() {
		t.Fatal()
	}

	if New(2, 3).IsScalar() {
		t.Fatal()
	}

}
func TestCount(t *testing.T) {

	v := New(1, 2, 2, 3)

	if v.Count(2) != 2 {
		t.Fatal()
	}

}
