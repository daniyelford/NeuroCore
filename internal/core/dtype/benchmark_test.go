package dtype

import "testing"

func BenchmarkName(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Float32.Name()
	}
}