package checkpoint

import (
	"testing"

	"github.com/daniyelford/neurocore/nn"
)

func TestSaveLoad(t *testing.T) {

	model :=
		nn.NewModel(
			nn.NewLinear(
				2,
				1,
			),
		)

	err :=
		Save(
			model,
			"test.json",
		)

	if err != nil {

		t.Fatal(err)

	}

	err =
		Load(
			model,
			"test.json",
		)

	if err != nil {

		t.Fatal(err)

	}

}
