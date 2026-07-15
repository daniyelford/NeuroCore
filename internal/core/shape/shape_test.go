package shape

import "testing"

func TestNew(t *testing.T) {

	s := New(3, 224, 224)

	if s.Rank() != 3 {
		t.Fatal("invalid rank")
	}

	if s.At(1) != 224 {
		t.Fatal("invalid dimension")
	}
}

func TestClone(t *testing.T) {

	s := New(2, 3)

	c := s.Clone()

	if !s.Equal(c) {
		t.Fatal("clone failed")
	}
}

func TestValid(t *testing.T) {

	if !New(2, 3).Valid() {
		t.Fatal("expected valid shape")
	}

	if New(2, 0).Valid() {
		t.Fatal("expected invalid shape")
	}
}

func TestGet(t *testing.T) {

	s := New(4, 5)

	v, ok := s.Get(1)

	if !ok {
		t.Fatal("expected valid index")
	}

	if v != 5 {
		t.Fatal("unexpected value")
	}

	_, ok = s.Get(100)

	if ok {
		t.Fatal("expected invalid index")
	}
}
func TestNumElements(t *testing.T) {

	s := New(3, 224, 224)

	if s.NumElements() != 150528 {
		t.Fatal()
	}

}

func TestScalar(t *testing.T) {

	if !New(1).IsScalar() {
		t.Fatal()
	}

	if New(10).IsScalar() {
		t.Fatal()
	}

}

func TestVector(t *testing.T) {

	if !New(100).IsVector() {
		t.Fatal()
	}

}

func TestMatrix(t *testing.T) {

	if !New(10, 20).IsMatrix() {
		t.Fatal()
	}

}

func TestTensor(t *testing.T) {

	if !New(3, 224, 224).IsTensor() {
		t.Fatal()
	}

}

func TestCanReshape(t *testing.T) {

	a := New(2, 3)

	b := New(6)

	if !a.CanReshape(b) {
		t.Fatal()
	}

}
func TestExpand(t *testing.T) {

	s := New(224, 224)

	n := s.Expand(0)

	if !n.Equal(New(1, 224, 224)) {
		t.Fatal()
	}

}

func TestExpandMiddle(t *testing.T) {

	s := New(10, 20)

	n := s.Expand(1)

	if !n.Equal(New(10, 1, 20)) {
		t.Fatal()
	}

}

func TestSqueeze(t *testing.T) {

	s := New(1, 3, 1, 224)

	n := s.Squeeze()

	if !n.Equal(New(3, 224)) {
		t.Fatal()
	}

}

func TestAppendShape(t *testing.T) {

	s := New(2, 3)

	n := s.Append(4, 5)

	if !n.Equal(New(2, 3, 4, 5)) {
		t.Fatal()
	}

}

func TestPrependShape(t *testing.T) {

	s := New(4, 5)

	n := s.Prepend(2, 3)

	if !n.Equal(New(2, 3, 4, 5)) {
		t.Fatal()
	}

}
func TestBroadcast1(t *testing.T) {

	a := New(3, 1)

	b := New(3, 5)

	if !a.CanBroadcast(b) {
		t.Fatal()
	}

	r, ok := a.BroadcastShape(b)

	if !ok {
		t.Fatal()
	}

	if !r.Equal(New(3, 5)) {
		t.Fatal()
	}

}

func TestBroadcast2(t *testing.T) {

	a := New(5)

	b := New(3, 5)

	if !a.CanBroadcast(b) {
		t.Fatal()
	}

	r, _ := a.BroadcastShape(b)

	if !r.Equal(New(3, 5)) {
		t.Fatal()
	}

}

func TestBroadcast3(t *testing.T) {

	a := New(2, 3)

	b := New(4, 5)

	if a.CanBroadcast(b) {
		t.Fatal()
	}

}
