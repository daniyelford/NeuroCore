package shape

import "testing"

func BenchmarkNumElements(b *testing.B) {

	s, _ := New(32, 64, 128)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		s.NumElements()
	}
}