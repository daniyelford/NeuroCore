package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func BenchmarkAccess(b *testing.B) {

	t := New(
		shape.New(100, 100),
	)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {

		_ = t.At(50, 50)

	}

}
