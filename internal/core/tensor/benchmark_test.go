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
