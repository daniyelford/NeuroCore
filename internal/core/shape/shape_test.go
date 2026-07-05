package shape

import "testing"

func TestShape(t *testing.T) {

	s, err := New(2, 3, 4)

	if err != nil {
		t.Fatal(err)
	}

	if s.Rank() != 3 {
		t.Fatal("rank should be 3")
	}

	if s.NumElements() != 24 {
		t.Fatal("num elements should be 24")
	}
}