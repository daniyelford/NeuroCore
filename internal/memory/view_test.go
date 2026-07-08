package memory

import "testing"

func TestView(t *testing.T) {

	m := From([]float32{
		1, 2, 3, 4, 5,
	})

	v := NewView(
		&m,
		1,
		3,
	)

	if v.Len() != 3 {

		t.Fatal()

	}

	if v.At(0) != 2 {

		t.Fatal()

	}

	v.Set(0, 100)

	if m.At(1) != 100 {

		t.Fatal("view is not connected")

	}

}
