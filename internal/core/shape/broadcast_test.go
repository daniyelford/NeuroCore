package shape

import "testing"

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
