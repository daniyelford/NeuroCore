package scalar

import "testing"

func BenchmarkAdd(b *testing.B) {

	x := New(5.0)
	y := New(10.0)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {

		x.Add(y)

	}

}