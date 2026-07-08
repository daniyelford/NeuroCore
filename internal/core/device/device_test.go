package device

import "testing"

func TestCPU(t *testing.T) {

	d := NewCPU()

	if d.Type() != CPU {

		t.Fatal()

	}

	if d.Name() != "cpu" {

		t.Fatal()

	}

}

func TestAllocate(t *testing.T) {

	d := NewCPU()

	m := d.Allocate(100)

	if m.Len() != 100 {

		t.Fatal()

	}

}
func TestRegistry(t *testing.T) {

	d, ok := Get("cpu")

	if !ok {

		t.Fatal()

	}

	if d.Name() != "cpu" {

		t.Fatal()

	}

}
