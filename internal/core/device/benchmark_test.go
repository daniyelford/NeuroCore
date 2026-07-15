package device

import "testing"

func BenchmarkAllocate(b *testing.B) {

	d := NewCPU()

	for b.Loop() {

		_ = d.Allocate(1024)

	}

}
