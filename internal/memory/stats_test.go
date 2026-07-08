package memory

import "testing"

func TestCapacity(t *testing.T) {

	m := New(100)

	if m.Capacity() != 100 {

		t.Fatal()

	}

}

func TestBytes(t *testing.T) {

	m := New(10)

	if m.Bytes() != 40 {

		t.Fatal()

	}

}
