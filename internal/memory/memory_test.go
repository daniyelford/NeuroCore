package memory

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/backend"
)

func TestNew(t *testing.T) {

	m := New(10)

	if m.Len() != 10 {

		t.Fatal()

	}

}

func TestSetGet(t *testing.T) {

	m := New(5)

	m.Set(2, 10.5)

	v, ok := m.Get(2)

	if !ok {

		t.Fatal()

	}

	if v != 10.5 {

		t.Fatal()

	}

}

func TestClone(t *testing.T) {

	m := From([]float32{1, 2, 3})

	c := m.Clone()

	if !m.Equal(c) {

		t.Fatal()

	}

	c.Set(0, 100)

	if m.At(0) == 100 {

		t.Fatal()

	}

}

func TestZero(t *testing.T) {

	m := From([]float32{1, 2, 3})

	m.Zero()

	if m.At(1) != 0 {

		t.Fatal()

	}

}
func TestFill(t *testing.T) {

	m := New(5)

	m.Fill(7)

	for i := 0; i < m.Len(); i++ {

		if m.At(i) != 7 {

			t.Fatal()

		}

	}

}
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
func TestMemoryDevice(t *testing.T) {

	m := New(10)

	if m.Device() != backend.CPU {

		t.Fatal()

	}

}
