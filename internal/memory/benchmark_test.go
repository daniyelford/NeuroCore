package memory

import "testing"

func BenchmarkClone(b *testing.B) {

	m := New(1024 * 1024)

	for b.Loop() {

		_ = m.Clone()

	}

}

func BenchmarkGet(b *testing.B) {

	m := New(1024)

	for b.Loop() {

		_, _ = m.Get(10)

	}

}
