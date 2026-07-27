package init

import (
	"math"
	"math/rand"

	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func (n Normal) Init(t *tensor.Tensor) {
	for i := 0; i < t.Len(); i++ {
		u1 := rand.Float32()
		u2 := rand.Float32()
		z := float32(math.Sqrt(-2*math.Log(float64(u1))) * math.Cos(2*math.Pi*float64(u2)))
		t.FlatSet(i, n.Mean+n.Std*z)
	}
}
func (k Kaiming) Init(t *tensor.Tensor) {
	shape := t.Shape().Values()
	if len(shape) != 2 {
		panic("kaiming requires 2D tensor")
	}
	fanIn := float32(shape[0])
	std := float32(math.Sqrt(2.0 / float64(fanIn)))
	Normal{Mean: 0, Std: std}.Init(t)
}
func (o Ones) Init(t *tensor.Tensor) {
	t.Fill(1)
}
func (u Uniform) Init(t *tensor.Tensor) {
	for i := 0; i < t.Len(); i++ {
		v := u.Min + rand.Float32()*(u.Max-u.Min)
		t.FlatSet(i, v)
	}
}
func (z Zeros) Init(t *tensor.Tensor) {
	t.Fill(0)
}
func (x Xavier) Init(t *tensor.Tensor) {
	shape := t.Shape().Values()
	if len(shape) != 2 {
		panic("xavier requires 2D tensor")
	}
	fanIn := float32(shape[0])
	fanOut := float32(shape[1])
	limit := float32(math.Sqrt(float64(6.0 / (fanIn + fanOut))))
	Uniform{Min: -limit, Max: limit}.Init(t)
}
