package errs

import "testing"

func TestErrors(t *testing.T) {

	if ErrInvalidArgument == nil {

		t.Fatal()

	}

}