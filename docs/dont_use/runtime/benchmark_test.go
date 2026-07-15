package runtime

import "testing"

func BenchmarkRuntimeNew(b *testing.B) {

	for b.Loop() {
		New()
	}

}
