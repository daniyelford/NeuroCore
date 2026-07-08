package tensor

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func BenchmarkAdd(
	b *testing.B,
) {

	a := New(
		shape.New(1000, 1000),
	)

	c := New(
		shape.New(1000, 1000),
	)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {

		_ = a.Add(c)

	}

}

func BenchmarkAddInplace(
	b *testing.B,
) {

	a := New(
		shape.New(1000, 1000),
	)

	c := New(
		shape.New(1000, 1000),
	)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {

		a.AddInplace(c)

	}

}
