package activation

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
	"github.com/daniyelford/neurocore/nn"
)

func NewReLU() *ReLU {
	return &ReLU{BaseModule: nn.NewBaseModule("ReLU")}
}
func (r *ReLU) Forward(input *autograd.Variable) (*autograd.Variable, error) {
	op := operations.NewReLU()
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
	return &Sigmoid{BaseModule: nn.NewBaseModule("Sigmoid")}
}
func (s *Sigmoid) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewSigmoid()
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
	return &Tanh{BaseModule: nn.NewBaseModule("Tanh")}
}
func (t *Tanh) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewTanh()
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
	return &LeakyReLU{BaseModule: nn.NewBaseModule("LeakyReLU"), Alpha: alpha}
}
func (l *LeakyReLU) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewLeakyReLU(l.Alpha)
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
	return &ELU{BaseModule: nn.NewBaseModule("ELU"), Alpha: alpha}
}
func (e *ELU) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewELU(e.Alpha)
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
	return &GELU{BaseModule: nn.NewBaseModule("GELU")}
}
func (g *GELU) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewGELU()
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
	op := operations.NewSoftplus()
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
	op := operations.NewSwish()
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
	op := operations.NewMish()
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
	op := operations.NewHardSigmoid()
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
	op := operations.NewHardSwish()
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
	op := operations.NewSoftmax(s.Axis)
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
	op := operations.NewLogSoftmax(l.Axis)
	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}
	return out
}
func NewL1Loss() *L1Loss {
	return &L1Loss{
		BaseModule: nn.NewBaseModule("L1Loss"),
	}
}
func (l *L1Loss) Forward(
	prediction,
	target *autograd.Variable,
) *autograd.Variable {

	op := operations.NewL1()

	out, err := op.Forward(
		prediction,
		target,
	)

	if err != nil {
		panic(err)
	}

	return out
}
func (l *L1Loss) Parameters() []nn.Parameter {
	return nil
}

func (l *L1Loss) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewSmoothL1Loss(beta float32) *SmoothL1Loss {
	return &SmoothL1Loss{
		BaseModule: nn.NewBaseModule("SmoothL1Loss"),
		Beta:       beta,
	}
}

func (l *SmoothL1Loss) Forward(
	pred,
	target *autograd.Variable,
) *autograd.Variable {

	op := operations.NewSmoothL1(l.Beta)

	out, err := op.Forward(pred, target)
	if err != nil {
		panic(err)
	}

	return out
}

func (l *SmoothL1Loss) Parameters() []nn.Parameter {
	return nil
}

func (l *SmoothL1Loss) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewHuberLoss(delta float32) *HuberLoss {
	return &HuberLoss{
		BaseModule: nn.NewBaseModule("HuberLoss"),
		Delta:      delta,
	}
}

func (l *HuberLoss) Forward(
	pred,
	target *autograd.Variable,
) *autograd.Variable {

	op := operations.NewHuber(l.Delta)

	out, err := op.Forward(
		pred,
		target,
	)

	if err != nil {
		panic(err)
	}

	return out
}

func (l *HuberLoss) Parameters() []nn.Parameter {
	return nil
}

func (l *HuberLoss) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewBCELoss() *BCELoss {

	return &BCELoss{
		BaseModule: nn.NewBaseModule("BCELoss"),
	}
}
func (l *BCELoss) Forward(
	pred,
	target *autograd.Variable,
) *autograd.Variable {

	op := operations.NewBCE()

	out, err := op.Forward(
		pred,
		target,
	)

	if err != nil {
		panic(err)
	}

	return out
}
func (l *BCELoss) Parameters() []nn.Parameter {
	return nil
}

func (l *BCELoss) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}

func NewBCEWithLogitsLoss() *BCEWithLogitsLoss {

	return &BCEWithLogitsLoss{
		BaseModule: nn.NewBaseModule("BCEWithLogitsLoss"),
	}
}

func (l *BCEWithLogitsLoss) Forward(
	pred,
	target *autograd.Variable,
) *autograd.Variable {
	op := operations.NewBCEWithLogits()
	out, err := op.Forward(
		pred,
		target,
	)
	if err != nil {
		panic(err)
	}
	return out
}
func (l *BCEWithLogitsLoss) Parameters() []nn.Parameter {
	return nil
}
func (l *BCEWithLogitsLoss) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
