package dtype

import "testing"

func BenchmarkName(b *testing.B) {
	for b.Loop() {
		_ = Float32.Name()
	}
}
