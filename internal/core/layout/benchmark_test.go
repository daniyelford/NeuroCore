package layout

import "testing"

func BenchmarkString(b *testing.B) {

	for i := 0; i < b.N; i++ {
		_ = RowMajor.String()
	}

}

func BenchmarkParse(b *testing.B) {

	for i := 0; i < b.N; i++ {

		_, _ = Parse("row-major")

	}

}
