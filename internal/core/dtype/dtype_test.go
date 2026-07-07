package dtype

import "testing"

func TestLookup(t *testing.T) {

	d, ok := ByName("float32")

	if !ok {
		t.Fatal("dtype not found")
	}

	if d != Float32 {
		t.Fatal("invalid dtype")
	}

}

func TestNumeric(t *testing.T) {

	if !Float64.IsNumeric() {
		t.Fatal()
	}

	if !Int32.IsNumeric() {
		t.Fatal()
	}

	if Bool.IsNumeric() {
		t.Fatal()
	}

}

func TestKinds(t *testing.T) {

	if !Float32.IsFloat() {
		t.Fatal()
	}

	if !Int64.IsSigned() {
		t.Fatal()
	}

	if !Uint64.IsUnsigned() {
		t.Fatal()
	}

	if !Complex64.IsComplex() {
		t.Fatal()
	}

}
