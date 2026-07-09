package nn

import (
	"testing"
)

func TestModelParameters(
	t *testing.T,
) {

	layer :=
		NewLinear(
			3,
			2,
		)

	model :=
		NewModel(
			layer,
		)

	if len(model.Parameters()) != 2 {

		t.Fatal()

	}

}
