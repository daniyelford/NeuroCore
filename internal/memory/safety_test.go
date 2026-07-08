package memory

import "testing"

func TestResize(t *testing.T) {

	m := New(5)

	m.Set(0, 10)

	m.Resize(10)

	if m.Len() != 10 {

		t.Fatal()

	}

	if m.At(0) != 10 {

		t.Fatal()

	}

}

func TestTrySet(t *testing.T) {

	m := New(3)

	if !m.TrySet(1, 5) {

		t.Fatal()

	}

	if m.TrySet(10, 5) {

		t.Fatal()

	}

}

func TestTryAt(t *testing.T) {

	m := From([]float32{1, 2, 3})

	v, ok := m.TryAt(1)

	if !ok || v != 2 {

		t.Fatal()

	}

	_, ok = m.TryAt(10)

	if ok {

		t.Fatal()

	}

}
