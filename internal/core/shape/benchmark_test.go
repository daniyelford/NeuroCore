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
func BenchmarkBroadcast(b *testing.B) {

	a := New(32, 1, 224)

	c := New(32, 64, 224)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {

		_, _ = a.BroadcastShape(c)

	}

}
func BenchmarkSqueeze(b *testing.B) {

	s := New(1, 3, 1, 224, 1)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {

		_ = s.Squeeze()

	}

}

func BenchmarkExpand(b *testing.B) {

	s := New(224, 224)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {

		_ = s.Expand(0)

	}

}
