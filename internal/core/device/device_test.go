package device

import "testing"

func BenchmarkAllocate(b *testing.B) {

	d := NewCPU()

	for b.Loop() {

		_ = d.Allocate(1024)

	}

}
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
func TestContext(t *testing.T) {

	ctx := DefaultContext()

	if ctx.Name() != "cpu" {

		t.Fatal()

	}

	if ctx.Type() != CPU {

		t.Fatal()

	}

}
