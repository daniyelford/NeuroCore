package scalar

import "testing"

func TestAdd(t *testing.T) {

	a := New(5.0)
	b := New(10.0)

	c := a.Add(b)

	if c.Value() != 15 {

		t.Fatalf("expected 15 got %v", c.Value())

	}

}

func TestSub(t *testing.T) {

	a := New(20.0)
	b := New(5.0)

	c := a.Sub(b)

	if c.Value() != 15 {

		t.Fatalf("expected 15 got %v", c.Value())

	}

}

func TestMul(t *testing.T) {

	a := New(3.0)
	b := New(5.0)

	c := a.Mul(b)

	if c.Value() != 15 {

		t.Fatalf("expected 15 got %v", c.Value())

	}

}

func TestDiv(t *testing.T) {

	a := New(30.0)
	b := New(2.0)

	c := a.Div(b)

	if c.Value() != 15 {

		t.Fatalf("expected 15 got %v", c.Value())

	}

}