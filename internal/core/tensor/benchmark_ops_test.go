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

	for b.Loop() {

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

	for b.Loop() {
		a.AddInplace(c)
	}

}
