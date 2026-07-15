package layout

import "testing"

func BenchmarkString(b *testing.B) {

	for b.Loop() {
		_ = RowMajor.String()
	}

}

func BenchmarkParse(b *testing.B) {

	for b.Loop() {

		_, _ = Parse("row-major")

	}

}
