package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func BenchmarkAccess(b *testing.B) {

	t := New(
		shape.New(100, 100),
	)

	for b.Loop() {

		_ = t.At(50, 50)

	}

}
