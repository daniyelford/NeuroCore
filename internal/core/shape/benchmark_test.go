package shape

import "testing"

func BenchmarkClone(b *testing.B) {

	s := New(3, 224, 224)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = s.Clone()
	}
}

func BenchmarkEqual(b *testing.B) {

	a := New(3, 224, 224)
	c := New(3, 224, 224)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = a.Equal(c)
	}
}
func BenchmarkNumElements(b *testing.B) {

	s := New(3, 224, 224)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = s.NumElements()
	}
}
