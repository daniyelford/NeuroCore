/*
Package nn contains neural network layers.
*/
package nn

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/internal/operations"
	initial "github.com/daniyelford/neurocore/nn/init"
)

func NewParameter(v *autograd.Variable) Parameter {
	return Parameter{Value: v}
}
func (p Parameter) NumElements() int {
	return p.Value.Data().Shape().NumElements()
}
func SaveJSON(state StateDict, path string) error {
	out := map[string]jsonVariable{}
	for name, v := range state {
		data := make([]float32, v.Data().Len())
		for i := 0; i < v.Data().Len(); i++ {
			data[i] = v.Data().FlatAt(i)
		}
		out[name] = jsonVariable{
			Shape: v.Data().Shape().Values(),
			Data:  data,
		}
	}
	bytes, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, bytes, 0644)
}
func (s *Sequential) StateDict() map[string]*autograd.Variable {
	result := map[string]*autograd.Variable{}
	for index, layer := range s.Modules {
		for name, value := range layer.StateDict() {
			key := fmt.Sprintf("%d.%s", index, name)
			result[key] = value
		}
	}
	return result
}
func NewSequential(modules ...Module) *Sequential {
	return &Sequential{
		BaseModule: NewBaseModule("Sequential"),
		Modules:    modules,
	}
}
func (s *Sequential) Forward(input *autograd.Variable) *autograd.Variable {
	out := input
	for _, m := range s.Modules {
		out = m.Forward(out)
	}
	return out
}
func (s Sequential) Parameters() []Parameter {
	params := []Parameter{}
	for _, layer := range s.Modules {
		params = append(params, layer.Parameters()...)
	}
	return params
}
func (s *Sequential) Children() []Module {
	return s.Modules
}
func NewMSELoss() *MSELoss {
	return &MSELoss{BaseModule: NewBaseModule("MSELoss")}
}
func (m *MSELoss) Parameters() []Parameter {
	return nil
}
func (m *MSELoss) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func (m *MSELoss) Forward(prediction *autograd.Variable, target *autograd.Variable) *autograd.Variable {
	op := &operations.MSE{}
	out, err := op.Forward(prediction, target)
	if err != nil {
		panic(err)
	}
	return out
}
func NewModel(module Module) *Model {
	return &Model{
		module:   module,
		training: true,
	}
}
func (m *Model) Train() {
	m.training = true
	m.module.Train()
}
func (m *Model) Eval() {
	m.training = false
	m.module.Eval()
}
func (m *Model) Parameters() []Parameter {
	return m.module.Parameters()
}
func (m *Model) Forward(input *autograd.Variable) *autograd.Variable {
	return m.module.Forward(input)
}
func (m *Model) StateDict() map[string]*autograd.Variable {
	return m.module.StateDict()
}
func (m *Model) LoadStateDict(state map[string]*autograd.Variable) error {
	current := m.StateDict()
	for name, value := range state {
		dst, ok := current[name]
		if !ok {
			return fmt.Errorf("unknown parameter: %s", name)
		}
		dst.SetData(value.Data().Clone())
		dst.SetRequiresGrad(value.RequiresGrad())
	}
	return nil
}
func NewMaxPool2D(kernelH, kernelW, strideH, strideW int) *MaxPool2D {
	return &MaxPool2D{BaseModule: NewBaseModule("MaxPool2D"), KernelH: kernelH, KernelW: kernelW, StrideH: strideH, StrideW: strideW}
}
func (m *MaxPool2D) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewMaxPool2D(m.KernelH, m.KernelW, m.StrideH, m.StrideW)
	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}
	return out
}
func (m *MaxPool2D) Parameters() []Parameter {
	return []Parameter{}
}
func (m *MaxPool2D) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func LoadJSON(path string) (StateDict, error) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	raw := map[string]jsonVariable{}
	err = json.Unmarshal(bytes, &raw)
	if err != nil {
		return nil, err
	}
	state := StateDict{}
	for name, j := range raw {
		t := tensor.New(shape.New(j.Shape...))
		for i, v := range j.Data {
			t.FlatSet(i, v)
		}
		state[name] = autograd.NewVariable(t, true)
	}
	return state, nil
}
func (l *Linear) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{
		"weight": l.Weight.Value,
		"bias":   l.Bias.Value,
	}
}
func NewLinear(in int, out int) *Linear {
	w := tensor.New(shape.New(in, out))
	initial.Xavier{}.Init(&w)
	b := tensor.New(shape.New(out))
	return &Linear{
		BaseModule: NewBaseModule("Linear"),
		Weight:     NewParameter(autograd.NewVariable(w, true)),
		Bias:       NewParameter(autograd.NewVariable(b, true)),
		In:         in,
		Out:        out,
	}
}
func (l *Linear) Forward(input *autograd.Variable) *autograd.Variable {
	matmul := &operations.MatMul{}
	x, err := matmul.Forward(input, l.Weight.Value)
	if err != nil {
		panic(err)
	}
	add := &operations.Add{}
	out, err := add.Forward(x, l.Bias.Value)
	if err != nil {
		panic(err)
	}
	return out
}
func (l *Linear) Parameters() []Parameter {
	return []Parameter{l.Weight, l.Bias}
}
func (l *Linear) Children() []Module {
	return nil
}
func NewFlatten(startDim int) *Flatten {
	return &Flatten{
		BaseModule: NewBaseModule("Flatten"),
		StartDim:   startDim,
	}
}
func (f *Flatten) Parameters() []Parameter {
	return nil
}
func (f *Flatten) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func (f *Flatten) Forward(input *autograd.Variable) *autograd.Variable {

	op := operations.NewFlatten(f.StartDim)

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}

//	func (f *Flatten) Forward(input *autograd.Variable) *autograd.Variable {
//		d := input.Data().Shape().Values()
//		if len(d) < 2 {
//			panic("flatten requires batch dimension")
//		}
//		batch := d[0]
//		size := 1
//		for i := 1; i < len(d); i++ {
//			size *= d[i]
//		}
//		out, ok := input.Data().Reshape(shape.New(batch, size))
//		if !ok {
//			panic("flatten reshape failed")
//		}
//		return autograd.NewVariable(out, input.RequiresGrad())
//	}
func NewEmbedding(numEmbeddings int, embeddingDim int) *Embedding {
	w := tensor.New(shape.New(numEmbeddings, embeddingDim))
	return &Embedding{
		BaseModule:    NewBaseModule("Embedding"),
		NumEmbeddings: numEmbeddings,
		EmbeddingDim:  embeddingDim,
		Weight:        NewParameter(autograd.NewVariable(w, true)),
	}
}
func (e *Embedding) Parameters() []Parameter {
	return []Parameter{e.Weight}
}
func (e *Embedding) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{"weight": e.Weight.Value}
}
func (e *Embedding) Forward(input *autograd.Variable) *autograd.Variable {
	indices, err := input.Data().Indices()
	if err != nil {
		panic(err)
	}
	out := tensor.New(shape.New(len(indices), e.EmbeddingDim))
	for i, index := range indices {
		if index < 0 || index >= e.NumEmbeddings {
			panic("embedding index out of range")
		}
		for j := 0; j < e.EmbeddingDim; j++ {
			v := e.Weight.Value.Data().At(index, j)
			out.Set(v, i, j)
		}
	}
	return autograd.NewVariable(out, input.RequiresGrad())
}
func NewDropout(p float32) *Dropout {
	return &Dropout{BaseModule: NewBaseModule("Dropout"), P: p, Training: true}
}
func (d *Dropout) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewDropout(d.P, d.Training)
	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}
	return out
}
func (d *Dropout) Parameters() []Parameter {
	return nil
}
func (d Dropout) Children() []Module {
	return nil
}
func (d *Dropout) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func (d *Dropout) Train() {
	d.Training = true
}
func (d *Dropout) Eval() {
	d.Training = false
}
func NewCrossEntropyLoss() *CrossEntropyLoss {
	return &CrossEntropyLoss{BaseModule: NewBaseModule("CrossEntropyLoss")}
}
func (c *CrossEntropyLoss) Parameters() []Parameter {
	return []Parameter{}
}
func (c *CrossEntropyLoss) Forward(input *autograd.Variable, target *autograd.Variable) *autograd.Variable {
	out, err := c.op.Forward(input, target)
	if err != nil {
		panic(err)
	}
	return out
}
func NewBatchNorm(numFeatures int) *BatchNorm {
	gamma := tensor.New(shape.New(numFeatures))
	gamma.Fill(1)
	beta := tensor.New(shape.New(numFeatures))
	runningMean := tensor.New(shape.New(numFeatures))
	runningVar := tensor.New(shape.New(numFeatures))
	runningVar.Fill(1)
	return &BatchNorm{
		BaseModule:  NewBaseModule("BatchNorm"),
		NumFeatures: numFeatures,
		Eps:         1e-5,
		Momentum:    0.1,
		Weight:      NewParameter(autograd.NewVariable(gamma, true)),
		Bias:        NewParameter(autograd.NewVariable(beta, true)),
		RunningMean: runningMean,
		RunningVar:  runningVar,
	}
}
func (b *BatchNorm) Parameters() []Parameter {
	return []Parameter{b.Weight, b.Bias}
}
func (b *BatchNorm) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{"weight": b.Weight.Value, "bias": b.Bias.Value}
}
func (b *BatchNorm) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewBatchNorm(b.NumFeatures, b.Eps)
	out, err := op.Forward(input, b.Weight.Value, b.Bias.Value)
	if err != nil {
		panic(err)
	}
	return out
}
func NewBatchNorm2D(channels int) *BatchNorm2D {
	gamma := tensor.New(shape.New(channels))
	beta := tensor.New(shape.New(channels))
	return &BatchNorm2D{
		BaseModule:  NewBaseModule("BatchNorm2D"),
		Gamma:       NewParameter(autograd.NewVariable(gamma, true)),
		Beta:        NewParameter(autograd.NewVariable(beta, true)),
		RunningMean: tensor.New(shape.New(channels)),
		RunningVar:  tensor.New(shape.New(channels)),
		Channels:    channels,
		Eps:         1e-5,
		Momentum:    0.1,
	}
}
func (b *BatchNorm2D) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewBatchNorm(b.Channels, b.Eps)
	out, err := op.Forward(input, b.Gamma.Value, b.Beta.Value)
	if err != nil {
		panic(err)
	}
	return out
}
func (b *BatchNorm2D) Parameters() []Parameter {
	return []Parameter{
		b.Gamma,
		b.Beta,
	}
}
func (b *BatchNorm2D) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{"gamma": b.Gamma.Value, "beta": b.Beta.Value}
}
func NewBaseModule(name string) BaseModule {
	return BaseModule{training: true, name: name}
}
func (b *BaseModule) Name() string {
	return b.name
}
func (b *BaseModule) Train() {
	b.training = true
	for _, child := range b.Children() {
		child.Train()
	}
}
func (b *BaseModule) Eval() {
	b.training = false
	for _, child := range b.Children() {
		child.Eval()
	}
}
func (b BaseModule) Training() bool {
	return b.training
}
func (b BaseModule) Children() []Module {
	return nil
}
func LayerNormNew(features int) LayerNorm {
	gamma := tensor.New(shape.New(features))
	beta := tensor.New(shape.New(features))
	return LayerNorm{
		BaseModule: NewBaseModule("LayerNorm"),
		Shape:      shape.New(features),
		Gamma:      NewParameter(autograd.NewVariable(gamma, true)),
		Beta:       NewParameter(autograd.NewVariable(beta, true)),
		Eps:        1e-5,
	}
}
func (l *LayerNorm) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewLayerNorm(l.Eps)
	out, err := op.Forward(input, l.Gamma.Value, l.Beta.Value)
	if err != nil {
		panic(err)
	}
	return out
}
func (l LayerNorm) Parameters() []Parameter {
	return []Parameter{l.Gamma, l.Beta}
}
func NewConv2D(inChannels int, outChannels int, kernelH int, kernelW int) *Conv2D {
	w := tensor.New(shape.New(outChannels, inChannels, kernelH, kernelW))
	b := tensor.New(shape.New(outChannels))
	return &Conv2D{
		BaseModule:  NewBaseModule("Conv2D"),
		Weight:      NewParameter(autograd.NewVariable(w, true)),
		Bias:        NewParameter(autograd.NewVariable(b, true)),
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelH:     kernelH,
		KernelW:     kernelW,
		Stride:      1,
		Padding:     0,
	}
}
func (c *Conv2D) Parameters() []Parameter {
	return []Parameter{c.Weight, c.Bias}
}
func (c *Conv2D) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{"weight": c.Weight.Value, "bias": c.Bias.Value}
}
func (c *Conv2D) Forward(input *autograd.Variable) *autograd.Variable {
	x := input.Data()
	d := x.Shape().Values()
	if len(d) != 4 {
		panic("Conv2D expects [N,C,H,W]")
	}
	n := d[0]
	h := d[2]
	w := d[3]
	outH := (h+2*c.Padding-c.KernelH)/c.Stride + 1
	outW := (w+2*c.Padding-c.KernelW)/c.Stride + 1
	out := tensor.New(shape.New(n, c.OutChannels, outH, outW))
	for b := range n {
		for oc := range c.OutChannels {
			for oy := range outH {
				for ox := range outW {
					var sum float32
					for ic := range c.InChannels {
						for ky := range c.KernelH {
							for kx := range c.KernelW {
								iy := oy*c.Stride + ky - c.Padding
								ix := ox*c.Stride + kx - c.Padding
								if iy < 0 || ix < 0 || iy >= h || ix >= w {
									continue
								}
								sum += x.At(b, ic, iy, ix) * c.Weight.Value.Data().At(oc, ic, ky, kx)
							}
						}
					}
					sum += c.Bias.Value.Data().At(oc)
					out.Set(sum, b, oc, oy, ox)
				}
			}
		}
	}
	return autograd.NewVariable(out, true)
}
func NewAvgPool2D(
	kernelH,
	kernelW,
	strideH,
	strideW int,
) *AvgPool2D {

	return &AvgPool2D{
		BaseModule: NewBaseModule("AvgPool2D"),
		KernelH:    kernelH,
		KernelW:    kernelW,
		StrideH:    strideH,
		StrideW:    strideW,
	}
}

func (m *AvgPool2D) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	op := operations.NewAvgPool2D(
		m.KernelH,
		m.KernelW,
		m.StrideH,
		m.StrideW,
	)

	out, err := op.Forward(input)

	if err != nil {
		panic(err)
	}

	return out
}

func (m *AvgPool2D) Parameters() []Parameter {
	return nil
}

func (m *AvgPool2D) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewAdaptiveAvgPool2D(
	outH,
	outW int,
) *AdaptiveAvgPool2D {

	return &AdaptiveAvgPool2D{
		BaseModule: NewBaseModule("AdaptiveAvgPool2D"),
		OutputH:    outH,
		OutputW:    outW,
	}
}
func (m *AdaptiveAvgPool2D) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	op := operations.NewAdaptiveAvgPool2D(
		m.OutputH,
		m.OutputW,
	)

	out, err := op.Forward(input)

	if err != nil {
		panic(err)
	}

	return out
}
func (m *AdaptiveAvgPool2D) Parameters() []Parameter {
	return nil
}
func (m *AdaptiveAvgPool2D) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewAdaptiveMaxPool2D(
	outputH,
	outputW int,
) *AdaptiveMaxPool2D {

	return &AdaptiveMaxPool2D{
		BaseModule: NewBaseModule("AdaptiveMaxPool2D"),
		OutputH:    outputH,
		OutputW:    outputW,
	}
}
func (m *AdaptiveMaxPool2D) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	op := operations.NewAdaptiveMaxPool2D(
		m.OutputH,
		m.OutputW,
	)

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}
func (m *AdaptiveMaxPool2D) Parameters() []Parameter {
	return nil
}
func (m *AdaptiveMaxPool2D) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewConvTranspose2D(
	inChannels,
	outChannels,
	kernelH,
	kernelW,
	strideH,
	strideW,
	paddingH,
	paddingW int,
) *ConvTranspose2D {

	// برای ConvTranspose وزن برعکس Conv2D است:
	// [inChannels, outChannels, kernelH, kernelW]

	w := tensor.New(
		shape.New(
			inChannels,
			outChannels,
			kernelH,
			kernelW,
		),
	)

	b := tensor.New(
		shape.New(outChannels),
	)

	return &ConvTranspose2D{
		BaseModule: NewBaseModule("ConvTranspose2D"),

		Weight: NewParameter(
			autograd.NewVariable(w, true),
		),

		Bias: NewParameter(
			autograd.NewVariable(b, true),
		),

		InChannels:  inChannels,
		OutChannels: outChannels,

		KernelH: kernelH,
		KernelW: kernelW,

		StrideH:  strideH,
		StrideW:  strideW,
		PaddingH: paddingH,
		PaddingW: paddingW,
	}
}
func (c *ConvTranspose2D) Forward(input *autograd.Variable) *autograd.Variable {
	op := operations.NewConvTranspose2D(
		c.StrideH,
		c.StrideW,
		c.PaddingH,
		c.PaddingW,
		c.KernelH,
		c.KernelW,
	)

	out, err := op.Forward(
		input,
		c.Weight.Value,
		c.Bias.Value,
	)
	if err != nil {
		panic(err)
	}

	return out
}
func (c *ConvTranspose2D) Parameters() []Parameter {
	return []Parameter{
		c.Weight,
		c.Bias,
	}
}
func (c *ConvTranspose2D) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{
		"weight": c.Weight.Value,
		"bias":   c.Bias.Value,
	}
}
func NewReflectionPad2D(
	top,
	bottom,
	left,
	right int,
) *ReflectionPad2D {

	return &ReflectionPad2D{
		BaseModule: NewBaseModule("ReflectionPad2D"),

		PadTop:    top,
		PadBottom: bottom,
		PadLeft:   left,
		PadRight:  right,
	}
}
func (r *ReflectionPad2D) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	op := operations.NewReflectionPad2D(
		r.PadTop,
		r.PadBottom,
		r.PadLeft,
		r.PadRight,
	)

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}
func (r *ReflectionPad2D) Parameters() []Parameter {
	return nil
}
func (r *ReflectionPad2D) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}

func NewReplicationPad2D(
	left,
	right,
	top,
	bottom int,
) *ReplicationPad2D {

	return &ReplicationPad2D{
		BaseModule: NewBaseModule("ReplicationPad2D"),
		Left:       left,
		Right:      right,
		Top:        top,
		Bottom:     bottom,
	}
}

func (r *ReplicationPad2D) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	op := operations.NewReplicationPad2D(
		r.Left,
		r.Right,
		r.Top,
		r.Bottom,
	)

	out, err := op.Forward(input)
	if err != nil {
		panic(err)
	}

	return out
}

func (r *ReplicationPad2D) Parameters() []Parameter {
	return nil
}

func (r *ReplicationPad2D) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewPixelShuffle(scale int) *PixelShuffle {
	return &PixelShuffle{
		BaseModule: NewBaseModule("PixelShuffle"),
		Scale:      scale,
	}
}

func (p *PixelShuffle) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	op := operations.NewPixelShuffle(p.Scale)

	out, err := op.Forward(input)

	if err != nil {
		panic(err)
	}

	return out
}

func (p *PixelShuffle) Parameters() []Parameter {
	return nil
}

func (p *PixelShuffle) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}

func NewPixelUnshuffle(scale int) *PixelUnshuffle {
	return &PixelUnshuffle{
		BaseModule: NewBaseModule("PixelUnshuffle"),
		Scale:      scale,
	}
}

func (p *PixelUnshuffle) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	op := operations.NewPixelUnshuffle(p.Scale)

	out, err := op.Forward(input)

	if err != nil {
		panic(err)
	}

	return out
}

func (p *PixelUnshuffle) Parameters() []Parameter {
	return nil
}

func (p *PixelUnshuffle) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}

//	func NewDropout(p float32) *Dropout {
//		return DropoutNew(p)
//	}
//	func (d *Dropout) Forward(input *autograd.Variable) *autograd.Variable {
//		if !d.Training() {
//			return input
//		}
//		data := input.Data()
//		out := data.Clone()
//		scale := float32(1.0)
//		if d.Probability < 1 {
//			scale = 1 / (1 - d.Probability)
//		}
//		for i := 0; i < out.Len(); i++ {
//			if rand.Float32() < d.Probability {
//				out.FlatSet(i, 0)
//			} else {
//				out.FlatSet(i, out.FlatAt(i)*scale)
//			}
//		}
//		return autograd.NewVariable(out, input.RequiresGrad())
//	}
//	func (l LayerNorm) Forward(input *autograd.Variable) *autograd.Variable {
//		x := input.Data()
//		var mean float32
//		for i := 0; i < x.Len(); i++ {
//			mean += x.FlatAt(i)
//		}
//		mean /= float32(x.Len())
//		var variance float32
//		for i := 0; i < x.Len(); i++ {
//			diff := x.FlatAt(i) - mean
//			variance += diff * diff
//		}
//		variance /= float32(x.Len())
//		out := tensor.New(x.Shape())
//		for i := 0; i < x.Len(); i++ {
//			n := (x.FlatAt(i) - mean) / float32(math.Sqrt(float64(variance+l.Eps)))
//			v := n*l.Gamma.Value.Data().FlatAt(i) + l.Beta.Value.Data().FlatAt(i)
//			out.FlatSet(i, v)
//		}
//		return autograd.NewVariable(out, true)
//	}
// func NewConv2D(
// 	inChannels int,
// 	outChannels int,
// 	kernelH int,
// 	kernelW int,
// 	stride int,
// 	padding int,
// ) *Conv2D {

// 	weight :=
// 		tensor.New(
// 			shape.New(
// 				outChannels,
// 				inChannels,
// 				kernelH,
// 				kernelW,
// 			),
// 		)

// 	bias :=
// 		tensor.New(
// 			shape.New(
// 				outChannels,
// 			),
// 		)

// 	return &Conv2D{

// 		BaseModule: NewBaseModule(),

// 		InChannels: inChannels,

// 		OutChannels: outChannels,

// 		KernelH: kernelH,

// 		KernelW: kernelW,

// 		StrideH: stride,

// 		StrideW: stride,

// 		PaddingH: padding,

// 		PaddingW: padding,

// 		Weight: NewParameter(
// 			autograd.NewVariable(
// 				weight,
// 				true,
// 			),
// 		),

// 		Bias: NewParameter(
// 			autograd.NewVariable(
// 				bias,
// 				true,
// 			),
// 		),
// 	}
// }
// func (c *Conv2D) Name() string {

// 	return "Conv2D"

// }

// func (c *Conv2D) Parameters() []Parameter {

// 	return []Parameter{

// 		c.Weight,

// 		c.Bias,
// 	}
// }

// func (c *Conv2D) StateDict() map[string]*autograd.Variable {

// 	return map[string]*autograd.Variable{

// 		"weight": c.Weight.Value,

// 		"bias": c.Bias.Value,
// 	}
// }
// func (c *Conv2D) Forward(
// 	input *autograd.Variable,
// ) *autograd.Variable {

// 	op :=
// 		operations.NewConv2D(
// 			c.StrideH,
// 			c.StrideW,
// 			c.PaddingH,
// 			c.PaddingW,
// 			c.KernelH,
// 			c.KernelW,
// 		)

// 	out, err :=
// 		op.Forward(
// 			input,
// 			c.Weight.Value,
// 			c.Bias.Value,
// 		)

// 	if err != nil {
// 		panic(err)
// 	}

// 	return out
// }
// func serializeTensor(
// 	t tensor.Tensor,
// ) SerializedTensor {

// 	data := make(
// 		[]float32,
// 		t.Len(),
// 	)

// 	for i := 0; i < t.Len(); i++ {

// 		data[i] = t.FlatAt(i)

// 	}

// 	return SerializedTensor{

// 		Shape: t.Shape().Values(),

// 		Data: data,
// 	}

// }

// func deserializeTensor(
// 	s SerializedTensor,
// ) tensor.Tensor {

// 	out :=
// 		tensor.New(
// 			shape.New(
// 				s.Shape...,
// 			),
// 		)

// 	for i, v := range s.Data {

// 		out.FlatSet(
// 			i,
// 			v,
// 		)

// 	}

// 	return out

// }
// func (d Dropout) Forward(
// 	input *autograd.Variable,
// ) *autograd.Variable {

// 	// evaluation mode
// 	if !d.Training() {

// 		return input

// 	}

// 	mask :=
// 		tensor.New(
// 			input.Data().Shape(),
// 		)

// 	for i := 0; i < mask.Len(); i++ {

// 		if rand.Float32() > d.P {

// 			mask.FlatSet(
// 				i,
// 				1,
// 			)

// 		} else {

// 			mask.FlatSet(
// 				i,
// 				0,
// 			)

// 		}

// 	}

// 	out :=
// 		input.Data().Mul(
// 			mask,
// 		)

// 	return *autograd.NewVariable(
// 		out,
// 		input.RequiresGrad(),
// 	)

// }
// func (d Dropout) Parameters() []Parameter {

// 	return nil

// }

// func NewMaxPool2D(
// 	kernelH int,
// 	kernelW int,
// ) *MaxPool2D {

// 	return &MaxPool2D{

// 		BaseModule: NewBaseModule(),

// 		KernelH: kernelH,

// 		KernelW: kernelW,

// 		Stride: kernelH,
// 	}

// }
// func (m *MaxPool2D) Forward(
// 	input *autograd.Variable,
// ) *autograd.Variable {

// 	in := input.Data()

// 	d := in.Shape().Values()

// 	if len(d) != 4 {

// 		panic("MaxPool2D expects [N,C,H,W]")

// 	}

// 	n := d[0]
// 	c := d[1]
// 	h := d[2]
// 	w := d[3]

// 	outH :=
// 		(h-m.KernelH)/m.Stride + 1

// 	outW :=
// 		(w-m.KernelW)/m.Stride + 1

// 	out :=
// 		tensor.New(
// 			shape.New(
// 				n,
// 				c,
// 				outH,
// 				outW,
// 			),
// 		)

// 	for b := 0; b < n; b++ {

// 		for ch := 0; ch < c; ch++ {

// 			for i := 0; i < outH; i++ {

// 				for j := 0; j < outW; j++ {

// 					max :=
// 						float32(-1e30)

// 					for kh := 0; kh < m.KernelH; kh++ {

// 						for kw := 0; kw < m.KernelW; kw++ {

// 							v :=
// 								in.At(
// 									b,
// 									ch,
// 									i*m.Stride+kh,
// 									j*m.Stride+kw,
// 								)

// 							if v > max {

// 								max = v

// 							}

// 						}

// 					}

// 					out.Set(
// 						max,
// 						b,
// 						ch,
// 						i,
// 						j,
// 					)

// 				}

// 			}

// 		}

// 	}

// 	return *autograd.NewVariable(
// 		out,
// 		false,
// 	)

// }
