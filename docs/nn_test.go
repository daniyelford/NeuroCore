package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/internal/operations"
	"github.com/daniyelford/neurocore/nn"
)

func TestLinear(
	t *testing.T,
) {

	layer :=
		nn.NewLinear(
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
		nn.NewModel(
			nn.NewLinear(3, 2),
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

	l := nn.NewLinear(4, 3)

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
func TestModelParameters(
	t *testing.T,
) {

	layer :=
		nn.NewLinear(
			3,
			2,
		)

	model :=
		nn.NewModel(
			layer,
		)

	if len(model.Parameters()) != 2 {

		t.Fatal()

	}

}
func TestSequential(t *testing.T) {

	model := nn.NewSequential(
		nn.NewLinear(3, 4),
		nn.NewLinear(4, 2),
	)

	x := autograd.NewVariable(
		tensor.New(shape.New(1, 3)),
		false,
	)

	y := model.Forward(*x)

	if y.Data().Shape().Values()[1] != 2 {
		t.Fatal()
	}

}
func TestMaxPool2D(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				2,
				2,
			),
		)

	x.Set(1, 0, 0, 0, 0)
	x.Set(2, 0, 0, 0, 1)
	x.Set(3, 0, 0, 1, 0)
	x.Set(4, 0, 0, 1, 1)

	pool :=
		nn.NewMaxPool2D(
			2,
			2,
			1,
			2,
		)

	out :=
		pool.Forward(
			*autograd.NewVariable(
				x,
				false,
			),
		)

	if out.Data().FlatAt(0) != 4 {

		t.Fatalf(
			"expected 4 got %f",
			out.Data().FlatAt(0),
		)

	}

}
func TestLayerNorm(t *testing.T) {

	x :=
		tensor.New(
			shape.New(4),
		)

	x.Set(1, 0)
	x.Set(2, 1)
	x.Set(3, 2)
	x.Set(4, 3)

	v :=
		autograd.NewVariable(
			x,
			false,
		)

	l :=
		nn.LayerNormNew(4)

	out :=
		l.Forward(
			*v,
		)

	if out.Data().Len() != 4 {

		t.Fatal()

	}

}

// func TestSequentialConvParameters(t *testing.T) {
// 	conv := nn.NewConv2D(1, 1, 3, 3, 1, 1)
// 	model :=
// 		nn.NewSequential(
// 			conv,
// 			nn.NewLinear(
// 				4,
// 				1,
// 			),
// 		)

// 	params :=
// 		model.Parameters()

// 	if len(params) != 4 {

// 		t.Fatalf(
// 			"expected 4 parameters got %d",
// 			len(params),
// 		)

// 	}

// }
func TestFlatten(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				2,
				3,
				4,
			),
		)

	f :=
		nn.NewFlatten()

	out :=
		f.Forward(
			*autograd.NewVariable(
				x,
				false,
			),
		)

	s :=
		out.Data().Shape().Values()

	if s[0] != 2 || s[1] != 12 {

		t.Fatalf(
			"wrong shape %v",
			s,
		)

	}

}
func TestSequentialStateDict(t *testing.T) {

	// model :=
	// 	nn.NewSequential(
	// 		nn.NewConv2D(
	// 			1,
	// 			2,
	// 			3,
	// 			3,
	// 			4,
	// 			1,
	// 		),
	// 	)
	model := nn.NewSequential(
		nn.NewLinear(2, 64),
		nn.NewDropout(0.5),
		nn.NewLinear(64, 10),
	)

	state :=
		model.StateDict()

	expected :=
		[]string{
			"0.weight",
			"0.bias",
		}

	for _, key := range expected {

		if _, ok := state[key]; !ok {

			t.Fatalf(
				"missing state key %s",
				key,
			)

		}

	}

}
func TestEmbeddingShape(t *testing.T) {

	e :=
		nn.NewEmbedding(
			100,
			8,
		)

	s :=
		e.Weight.Value.Data().
			Shape().
			Values()

	if s[0] != 100 ||
		s[1] != 8 {

		t.Fatal(
			"wrong embedding shape",
		)

	}

}

func TestEmbeddingForward(t *testing.T) {

	input :=
		tensor.New(
			shape.New(3),
		)

	input.Set(1, 0)
	input.Set(5, 1)
	input.Set(9, 2)

	v :=
		autograd.NewVariable(
			input,
			false,
		)

	e :=
		nn.NewEmbedding(
			20,
			4,
		)

	out :=
		e.Forward(
			*v,
		)

	s :=
		out.Data().
			Shape().
			Values()

	if s[0] != 3 ||
		s[1] != 4 {

		t.Fatal(
			"invalid embedding output shape",
		)

	}

}

func TestEmbeddingLookup(t *testing.T) {

	e :=
		nn.NewEmbedding(
			10,
			3,
		)

	w :=
		e.Weight.Value.Data()

	w.Set(10, 2, 0)
	w.Set(20, 2, 1)
	w.Set(30, 2, 2)

	input :=
		tensor.New(
			shape.New(1),
		)

	input.Set(
		2,
		0,
	)

	out :=
		e.Forward(
			*autograd.NewVariable(
				input,
				false,
			),
		)

	if out.Data().At(0, 0) != 10 ||
		out.Data().At(0, 1) != 20 ||
		out.Data().At(0, 2) != 30 {

		t.Fatal(
			"lookup failed",
		)

	}

}
func TestDropoutParameters(t *testing.T) {

	d := nn.NewDropout(0.5)

	if len(d.Parameters()) != 0 {

		t.Fatal("dropout should not have parameters")

	}

}

func TestDropoutStateDict(t *testing.T) {

	d := nn.NewDropout(0.5)

	if len(d.StateDict()) != 0 {

		t.Fatal("dropout should not have state")

	}

}

// func TestDropoutEval(t *testing.T) {

// 	x := tensor.New(
// 		shape.New(4),
// 	)

// 	x.FlatSet(0, 1)
// 	x.FlatSet(1, 2)
// 	x.FlatSet(2, 3)
// 	x.FlatSet(3, 4)

// 	v := autograd.NewVariable(
// 		x,
// 		false,
// 	)

// 	d := nn.NewDropout(0.5)

// 	d.Eval()

// 	out := d.Forward(*v)

// 	for i := 0; i < x.Len(); i++ {

// 		if out.Data().FlatAt(i) != x.FlatAt(i) {

// 			t.Fatal("eval mode should not modify tensor")

// 		}

// 	}

// }

// func TestDropoutTrain(t *testing.T) {

// 	x := tensor.New(
// 		shape.New(100),
// 	)

// 	x.Fill(1)

// 	v := autograd.NewVariable(
// 		x,
// 		false,
// 	)

// 	d := nn.NewDropout(0.5)

// 	d.Train()

// 	out := d.Forward(*v)

// 	hasZero := false

// 	for i := 0; i < out.Data().Len(); i++ {

// 		if out.Data().FlatAt(i) == 0 {

// 			hasZero = true
// 			break

// 		}

// 	}

// 	if !hasZero {

// 		t.Fatal("expected at least one dropped element")

// 	}

// }
func TestCrossEntropyLoss(
	t *testing.T,
) {

	logitsTensor :=
		tensor.New(
			shape.New(
				2,
				3,
			),
		)

	// sample 0
	logitsTensor.Set(
		2,
		0,
		0,
	)

	logitsTensor.Set(
		1,
		0,
		1,
	)

	logitsTensor.Set(
		0,
		0,
		2,
	)

	// sample 1
	logitsTensor.Set(
		0,
		1,
		0,
	)

	logitsTensor.Set(
		1,
		1,
		1,
	)

	logitsTensor.Set(
		2,
		1,
		2,
	)

	logits :=
		autograd.NewVariable(
			logitsTensor,
			true,
		)

	targetTensor :=
		tensor.New(
			shape.New(2),
		)

	targetTensor.Set(
		0,
		0,
	)

	targetTensor.Set(
		2,
		1,
	)

	target :=
		autograd.NewVariable(
			targetTensor,
			false,
		)

	lossFn :=
		nn.NewCrossEntropyLoss()

	out :=
		lossFn.Forward(
			*logits,
			*target,
		)

	if out.Data().NumElements() != 1 {

		t.Fatalf(
			"expected scalar loss",
		)

	}

	autograd.Backward(
		&out,
	)

	if logits.Grad().NumElements() != 6 {

		t.Fatalf(
			"invalid gradient shape",
		)

	}

}

func TestConv2DStride(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				6,
				6,
			),
		)

	conv :=
		nn.NewConv2D(
			1,
			1,
			3,
			3,
		)

	out :=
		conv.Forward(
			*autograd.NewVariable(
				x,
				false,
			),
		)

	s :=
		out.Data().
			Shape().
			Values()

	if s[2] != 2 ||
		s[3] != 2 {

		t.Fatal(
			"wrong stride output shape",
		)

	}

}
func TestConv2DSamePadding(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				5,
				5,
			),
		)

	conv :=
		nn.NewConv2D(
			1,
			1,
			3,
			3,
		)

	out :=
		conv.Forward(
			*autograd.NewVariable(
				x,
				false,
			),
		)

	s :=
		out.Data().
			Shape().
			Values()

	if s[2] != 5 ||
		s[3] != 5 {

		t.Fatal(
			"same padding failed",
		)

	}

}
func TestConv2DGraph(t *testing.T) {

	conv :=
		nn.NewConv2D(
			1,
			1,
			3,
			3,
		)

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				5,
				5,
			),
		)

	input :=
		autograd.NewVariable(
			x,
			true,
		)

	out :=
		conv.Forward(
			*input,
		)

	if out.Node().Op == nil {

		t.Fatal(
			"conv not connected to graph",
		)

	}

	if len(out.Node().Parents) != 3 {

		t.Fatal(
			"conv parents missing",
		)

	}

}

func TestBatchNormParameters(t *testing.T) {

	bn := nn.NewBatchNorm(4)

	if len(bn.Parameters()) != 2 {

		t.Fatal("batchnorm should have 2 parameters")

	}

}

func TestBatchNormStateDict(t *testing.T) {

	bn := nn.NewBatchNorm(4)

	s := bn.StateDict()

	if len(s) != 2 {

		t.Fatal("invalid state dict")

	}

}

// func TestBatchNormForwardShape(t *testing.T) {

// 	x := tensor.New(
// 		shape.New(2, 4),
// 	)

// 	x.Fill(1)

// 	v := autograd.NewVariable(
// 		x,
// 		false,
// 	)

// 	bn := nn.NewBatchNorm(4)

// 	out := bn.Forward(*v)

// 	if !out.Data().Shape().Equal(x.Shape()) {

// 		t.Fatal("shape mismatch")

// 	}

// }

func TestBatchNormGammaInitializedToOne(t *testing.T) {

	bn := nn.NewBatchNorm(4)

	for i := 0; i < 4; i++ {

		if bn.Weight.Value.Data().FlatAt(i) != 1 {

			t.Fatal("gamma should initialize to one")

		}

	}

}

func TestBatchNormRunningVarianceInitializedToOne(t *testing.T) {

	bn := nn.NewBatchNorm(4)

	for i := 0; i < 4; i++ {

		if bn.RunningVar.FlatAt(i) != 1 {

			t.Fatal("running variance should initialize to one")

		}

	}

}
func TestBackwardRoot(t *testing.T) {

	data := tensor.New(
		shape.New(1),
	)

	data.Set(
		2,
		0,
	)

	x := autograd.NewVariable(
		data,
		true,
	)

	autograd.Backward(x)

	if x.Grad().Len() != 1 {

		t.Fatal(
			"gradient not created",
		)

	}

}
func TestSimpleBackwardEngin(t *testing.T) {
	data := tensor.New(shape.New(1))
	data.Set(2, 0)
	x := autograd.NewVariable(data, true)
	engine := autograd.NewEngine()
	y, err := engine.Execute(new(operations.Add), x, x)
	if err != nil {
		t.Fatal(err)
	}
	autograd.Backward(y)
	if x.Grad().At(0) != 2 {
		t.Fatal(x.Grad().At(0))
	}
}

func TestDropoutTrain(t *testing.T) {

	x :=
		tensor.New(
			shape.New(10),
		)

	v :=
		autograd.NewVariable(
			x,
			false,
		)

	d :=
		nn.NewDropout(
			0.5,
		)

	out :=
		d.Forward(
			*v,
		)

	if out.Data().Len() != 10 {

		t.Fatal()

	}

}

func TestDropoutEval(t *testing.T) {

	x :=
		tensor.New(
			shape.New(5),
		)

	v :=
		autograd.NewVariable(
			x,
			false,
		)

	d :=
		nn.NewDropout(
			0.5,
		)

	d.Eval()

	out :=
		d.Forward(
			*v,
		)

	if !out.Data().Shape().Equal(
		x.Shape(),
	) {

		t.Fatal()

	}

}
func TestConv2DForward(t *testing.T) {

	conv :=
		nn.NewConv2D(
			1,
			1,
			1,
			1,
		)

	// weight = 1
	conv.Weight.Value.Data().Set(
		1,
		0,
		0,
		0,
		0,
	)

	// bias = 0
	conv.Bias.Value.Data().Set(
		0,
		0,
	)

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				3,
				3,
			),
		)

	value := float32(1)

	for i := 0; i < 3; i++ {

		for j := 0; j < 3; j++ {

			x.Set(
				value,
				0,
				0,
				i,
				j,
			)

			value++

		}

	}

	input :=
		autograd.NewVariable(
			x,
			false,
		)

	out :=
		conv.Forward(
			*input,
		)

	expected :=
		[]float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		}

	index := 0

	for i := 0; i < 3; i++ {

		for j := 0; j < 3; j++ {

			got :=
				out.Data().At(
					0,
					0,
					i,
					j,
				)

			if got != expected[index] {

				t.Fatalf(
					"wrong output at %d got %v expected %v",
					index,
					got,
					expected[index],
				)

			}

			index++

		}

	}

}
