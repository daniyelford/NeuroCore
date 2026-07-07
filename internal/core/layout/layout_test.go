package layout

import "testing"

func TestDefault(t *testing.T) {

	if Default() != RowMajor {
		t.Fatal()
	}

}

func TestValid(t *testing.T) {

	if !RowMajor.Valid() {
		t.Fatal()
	}

	if !ColumnMajor.Valid() {
		t.Fatal()
	}

	if Unknown.Valid() {
		t.Fatal()
	}

}

func TestParse(t *testing.T) {

	v, ok := Parse("row-major")

	if !ok {
		t.Fatal()
	}

	if v != RowMajor {
		t.Fatal()
	}

	v, ok = Parse("fortran")

	if !ok {
		t.Fatal()
	}

	if v != ColumnMajor {
		t.Fatal()
	}

}

func TestString(t *testing.T) {

	if RowMajor.String() != "row-major" {
		t.Fatal()
	}

}
