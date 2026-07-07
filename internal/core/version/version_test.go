package version

import "testing"

func TestString(t *testing.T) {

	if String() != "0.1.0" {

		t.Fatal("invalid version")

	}

}