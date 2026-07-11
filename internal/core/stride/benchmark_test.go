package stride

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/layout"
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func BenchmarkCompute(b *testing.B) {

	sh := shape.New(3, 224, 224)

	for b.Loop() {
		_ = Compute(sh, layout.RowMajor)
	}
}

func BenchmarkOffset(b *testing.B) {

	s := New(50176, 224, 1)

	for b.Loop() {
		_ = s.Offset(2, 10, 15)
	}
}

func BenchmarkClone(b *testing.B) {

	s := New(50176, 224, 1)

	for b.Loop() {
		_ = s.Clone()
	}
}
