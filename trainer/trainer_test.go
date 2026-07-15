package trainer

import (
	"testing"

	"github.com/daniyelford/neurocore/dataset"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/nn/optimizer"
)

func TestLinearDebug(t *testing.T) {

	l := nn.NewLinear(2, 1)

	println(
		"weight shape rank",
		l.Weight.Value.Data().Shape().Rank(),
	)

	println(
		"weight elements",
		l.Weight.Value.Data().Shape().NumElements(),
	)

	println(
		"bias elements",
		l.Bias.Value.Data().Shape().NumElements(),
	)
}
func TestShape(t *testing.T) {

	s := shape.New(2, 3)

	println(
		"rank",
		s.Rank(),
	)

	println(
		"elements",
		s.NumElements(),
	)

	for _, v := range s.Values() {
		println(v)
	}

}
func TestTrainerLinear(
	t *testing.T,
) {

	x :=
		tensor.New(
			shape.New(4, 1),
		)

	y :=
		tensor.New(
			shape.New(4, 1),
		)

	// y = 2x

	x.Set(1, 0, 0)
	x.Set(2, 1, 0)
	x.Set(3, 2, 0)
	x.Set(4, 3, 0)

	y.Set(2, 0, 0)
	y.Set(4, 1, 0)
	y.Set(6, 2, 0)
	y.Set(8, 3, 0)

	ds :=
		dataset.NewTensorDataset(
			x,
			y,
		)

	loader :=
		dataset.NewDataLoader(
			ds,
			4,
		)
	model :=
		nn.NewModel(nn.NewLinear(
			1,
			1,
		),
		)

	sgd :=
		optimizer.NewSGD(
			0.01,
		)

	tr :=
		NewTrainer(
			model,
			sgd,
		)
	tr.Train(
		loader,
		10,
		nn.MSELoss{}.,
	)

}
