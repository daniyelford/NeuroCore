package device

import "testing"

func TestContext(t *testing.T) {

	ctx := DefaultContext()

	if ctx.Name() != "cpu" {

		t.Fatal()

	}

	if ctx.Type() != CPU {

		t.Fatal()

	}

}
