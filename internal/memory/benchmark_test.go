package memory

import "testing"

func BenchmarkClone(b *testing.B) {

	m := New(1024 * 1024)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {

		_ = m.Clone()

	}

}

// func BenchmarkGet(b *testing.B) {

// 	m := New(1024)

// 	b.ResetTimer()

// 	for i := 0; i < b.N; i++ {

// 		_, _ = m.Get(10)

// 	}

// }
