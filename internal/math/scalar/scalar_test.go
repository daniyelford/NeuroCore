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