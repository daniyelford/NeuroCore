package activation

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
	"github.com/daniyelford/neurocore/nn"
)

func NewReLU() *ReLU {
	return &ReLU{
		BaseModule: nn.NewBaseModule("ReLU"),
	}
}
func (r *ReLU) Forward(input *autograd.Variable) (*autograd.Variable, error) {
	op := &operations.ReLU{}
	out, err := op.Forward(input)
	return out, err
}
func (r *ReLU) Parameters() []nn.Parameter {
	return nil
}
func (r *ReLU) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewSigmoid() *Sigmoid {
	return &Sigmoid{
		BaseModule: nn.NewBaseModule("Sigmoid"),
	}
}
func (s *Sigmoid) Forward(input *autograd.Variable) *autograd.Variable {
	op := &operations.Sigmoid{}
	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}
	return out
}
func (s *Sigmoid) Parameters() []nn.Parameter {
	return nil
}
func (s *Sigmoid) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewTanh() *Tanh {
	return &Tanh{
		BaseModule: nn.NewBaseModule("Tanh"),
	}
}
func (t *Tanh) Forward(input *autograd.Variable) *autograd.Variable {
	op := &operations.Tanh{}
	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}
	return out
}
func (t *Tanh) Parameters() []nn.Parameter {
	return nil
}
func (t *Tanh) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewLeakyReLU(alpha float32) *LeakyReLU {
	return &LeakyReLU{
		BaseModule: nn.NewBaseModule("LeakyReLU"),
		Alpha:      alpha,
	}
}
func (l *LeakyReLU) Forward(
	input *autograd.Variable,
) *autograd.Variable {
	op := operations.LeakyReLU{NegativeSlope: l.Alpha}
	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}
	return out
}
func (l *LeakyReLU) Parameters() []nn.Parameter {
	return nil
}
func (l *LeakyReLU) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewELU(alpha float32) *ELU {
	return &ELU{
		BaseModule: nn.NewBaseModule("ELU"),
		Alpha:      alpha,
	}
}
func (e *ELU) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.ELU{Alpha: e.Alpha}
	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}
	return out
}
func (e *ELU) Parameters() []nn.Parameter {
	return nil
}
func (e *ELU) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewGELU() *GELU {
	return &GELU{
		BaseModule: nn.NewBaseModule("GELU"),
	}
}
func (g *GELU) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.GELU{}
	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}
	return out
}
func (g *GELU) Parameters() []nn.Parameter {
	return nil
}
func (g *GELU) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewSoftplus() *Softplus {
	return &Softplus{BaseModule: nn.NewBaseModule("Softplus")}
}
func (s *Softplus) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.Softplus{}
	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}
	return out
}
func (s *Softplus) Parameters() []nn.Parameter {
	return nil
}
func (s *Softplus) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewSwish() *Swish {
	return &Swish{BaseModule: nn.NewBaseModule("Swish")}
}
func (s *Swish) Forward(input *autograd.Variable) *autograd.Variable {

	op := operations.Swish{}

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}
func (s *Swish) Parameters() []nn.Parameter {
	return nil
}
func (s *Swish) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewMish() *Mish {
	return &Mish{BaseModule: nn.NewBaseModule("Mish")}
}
func (m *Mish) Forward(input *autograd.Variable) *autograd.Variable {

	op := operations.Mish{}

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}
func (m *Mish) Parameters() []nn.Parameter {
	return nil
}
func (m *Mish) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewHardSigmoid() *HardSigmoid {
	return &HardSigmoid{BaseModule: nn.NewBaseModule("HardSigmoid")}
}
func (h *HardSigmoid) Forward(input *autograd.Variable) *autograd.Variable {

	op := operations.HardSigmoid{}

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}
func (h *HardSigmoid) Parameters() []nn.Parameter {
	return nil
}
func (h *HardSigmoid) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewHardSwish() *HardSwish {
	return &HardSwish{BaseModule: nn.NewBaseModule("HardSwish")}
}
func (h *HardSwish) Forward(input *autograd.Variable) *autograd.Variable {

	op := operations.HardSwish{}

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}
func (h *HardSwish) Parameters() []nn.Parameter {
	return nil
}
func (h *HardSwish) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewSoftmax(axis int) *Softmax {
	return &Softmax{BaseModule: nn.NewBaseModule("Softmax"), Axis: axis}
}
func (s *Softmax) Forward(input *autograd.Variable) *autograd.Variable {

	op := operations.Softmax{
		Axis: s.Axis,
	}

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}
func NewLogSoftmax(axis int) *LogSoftmax {
	return &LogSoftmax{BaseModule: nn.NewBaseModule("LogSoftmax"), Axis: axis}
}
func (l *LogSoftmax) Forward(input *autograd.Variable) *autograd.Variable {

	op := operations.LogSoftmax{
		Axis: l.Axis,
	}

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}
