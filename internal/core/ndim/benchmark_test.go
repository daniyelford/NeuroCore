package ndim

import "testing"

func BenchmarkClone(b *testing.B) {
	v := New(1, 2, 3, 4, 5, 6)
	for b.Loop() {
		_ = v.Clone()
	}
}

func BenchmarkSum(b *testing.B) {
	v := New(1, 2, 3, 4, 5, 6)
	b.ResetTimer()
	for b.Loop() {
		_ = v.Sum()
	}
}

func BenchmarkProduct(b *testing.B) {
	v := New(1, 2, 3, 4, 5)
	b.ResetTimer()
	for b.Loop() {
		_ = v.Product()
	}
}

func BenchmarkEqual(b *testing.B) {
	a := New(1, 2, 3, 4, 5)
	c := New(1, 2, 3, 4, 5)
	b.ResetTimer()
	for b.Loop() {
		_ = a.Equal(c)
	}
}

func BenchmarkValues(b *testing.B) {
	v := New(1, 2, 3, 4, 5)
	b.ResetTimer()
	for b.Loop() {
		_ = v.Values()
	}
}
