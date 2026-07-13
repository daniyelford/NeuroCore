package optim

// import (
// 	"testing"

// 	"github.com/daniyelford/neurocore/activation"
// 	"github.com/daniyelford/neurocore/internal/autograd"
// 	"github.com/daniyelford/neurocore/internal/core/shape"
// 	"github.com/daniyelford/neurocore/internal/core/tensor"
// )

// func TestReLU(t *testing.T) {

// 	x := tensor.From(
// 		shape.New(3),
// 		[]float32{
// 			-1, 2, -3,
// 		},
// 	)

// 	v := autograd.NewVariable(
// 		x,
// 		false,
// 	)

// 	r := activation.NewReLU()

// 	y := r.Forward(v)

// 	if y.Data().At(0) != 0 {

// 		t.Fatal()

// 	}

// 	if y.Data().At(1) != 2 {

// 		t.Fatal()

// 	}

// }
