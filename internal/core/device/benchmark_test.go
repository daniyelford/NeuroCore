package device

import "testing"

func BenchmarkAllocate(b *testing.B) {

	d := NewCPU()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {

		_ = d.Allocate(1024)

	}

}
