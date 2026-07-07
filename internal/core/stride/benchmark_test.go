package stride

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func BenchmarkCompute(b *testing.B) {

	sh := shape.New(3, 224, 224)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = Compute(sh)
	}
}

func BenchmarkOffset(b *testing.B) {

	s := New(50176, 224, 1)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = s.Offset(2, 10, 15)
	}
}

func BenchmarkClone(b *testing.B) {

	s := New(50176, 224, 1)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = s.Clone()
	}
}
