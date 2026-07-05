package dtype

import "testing"

func TestSize(t *testing.T) {
	if Float64.Size() != 8 {
		t.Fatal("float64 size should be 8")
	}

	if Int16.Size() != 2 {
		t.Fatal("int16 size should be 2")
	}
}

func TestParse(t *testing.T) {
	dt, err := Parse("float32")
	if err != nil {
		t.Fatal(err)
	}

	if dt != Float32 {
		t.Fatal("wrong dtype")
	}
}