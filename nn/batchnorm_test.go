package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestBatchNormParameters(t *testing.T) {

	bn := NewBatchNorm(4)

	if len(bn.Parameters()) != 2 {

		t.Fatal("batchnorm should have 2 parameters")

	}

}

func TestBatchNormStateDict(t *testing.T) {

	bn := NewBatchNorm(4)

	s := bn.StateDict()

	if len(s) != 2 {

		t.Fatal("invalid state dict")

	}

}

func TestBatchNormForwardShape(t *testing.T) {

	x := tensor.New(
		shape.New(2, 4),
	)

	x.Fill(1)

	v := autograd.NewVariable(
		x,
		false,
	)

	bn := NewBatchNorm(4)

	out := bn.Forward(*v)

	if !out.Data().Shape().Equal(x.Shape()) {

		t.Fatal("shape mismatch")

	}

}

func TestBatchNormGammaInitializedToOne(t *testing.T) {

	bn := NewBatchNorm(4)

	for i := 0; i < 4; i++ {

		if bn.Weight.Value.Data().FlatAt(i) != 1 {

			t.Fatal("gamma should initialize to one")

		}

	}

}

func TestBatchNormRunningVarianceInitializedToOne(t *testing.T) {

	bn := NewBatchNorm(4)

	for i := 0; i < 4; i++ {

		if bn.RunningVar.FlatAt(i) != 1 {

			t.Fatal("running variance should initialize to one")

		}

	}

}
