package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestLinear(
	t *testing.T,
) {

	layer :=
		NewLinear(
			3,
			2,
		)

	x := *autograd.NewVariable(
		tensor.New(
			shape.New(1, 3),
		),
		false,
	)

	y := layer.Forward(x)

	if y.Data().Shape().Values()[1] != 2 {

		t.Fatal()

	}

}

func TestStateDict(t *testing.T) {

	model :=
		NewModel(
			NewLinear(3, 2),
		)

	state :=
		model.StateDict()

	if len(state) != 2 {

		t.Fatal(
			"wrong state size",
		)

	}

	_, ok :=
		state["weight"]

	if !ok {

		t.Fatal(
			"weight missing",
		)

	}

}
func TestLinearInitialization(t *testing.T) {

	l := NewLinear(4, 3)

	allZero := true

	for i := 0; i < l.Weight.Value.Data().Len(); i++ {

		if l.Weight.Value.Data().FlatAt(i) != 0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Fatal("linear weight is not initialized")
	}

}

// func TestSaveLoadJSON(
// 	t *testing.T,
// ) {

// 	model :=
// 		NewModel(
// 			NewLinear(
// 				2,
// 				1,
// 			),
// 		)

// 	err :=
// 		SaveJSON(
// 			"model.json",
// 		)

// 	if err != nil {

// 		t.Fatal(err)

// 	}

// 	model2 :=
// 		NewModel(
// 			NewLinear(
// 				2,
// 				1,
// 			),
// 		)

// 	err =
// 		model2.LoadJSON(
// 			"model.json",
// 		)

// 	if err != nil {

// 		t.Fatal(err)

// 	}

// 	a :=
// 		model.StateDict()

// 	b :=
// 		model2.StateDict()

// 	for name, v := range a {

// 		if !v.Data().Equal(
// 			b[name].Data(),
// 		) {

// 			t.Fatal(name)

// 		}

// 	}

// }
