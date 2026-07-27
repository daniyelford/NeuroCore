package operations

import (
	"errors"
	"math"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func (b *Base) Save(inputs ...*autograd.Variable) {
	b.inputs = append(b.inputs[:0], inputs...)
}
func (b *Base) Inputs() []*autograd.Variable {
	return b.inputs
}
func (b *Base) Input(i int) *autograd.Variable {
	return b.inputs[i]
}
func (b *Base) Output() *autograd.Variable {
	return b.output
}
func (b *Base) SetOutput(v *autograd.Variable) {
	b.output = v
}
func (op *Add) Name() string {
	return "Add"
}
func (op *Add) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 2 {
		return nil, errors.New("add requires 2 inputs")
	}
	a := inputs[0]
	b := inputs[1]
	op.Save(a, b)
	out := autograd.NewVariable(a.Data().Add(b.Data()), a.RequiresGrad() || b.RequiresGrad())
	out.Node().Parents = []*autograd.Node{a.Node(), b.Node()}
	out.Node().Op = op
	op.SetOutput(out)
	return out, nil
}
func (op *Add) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	a := op.Input(0).Data()
	b := op.Input(1).Data()
	gradA := grad
	gradB := grad
	if !a.Shape().Equal(b.Shape()) {
		if len(b.Shape().Values()) == 1 {
			gradB = grad.ReduceSumAxis(0)
		}
	}
	return []tensor.Tensor{gradA, gradB}, nil
}
func (op *Transpose) Name() string {
	return "Transpose"
}
func (op *Transpose) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("transpose requires exactly 1 input")
	}
	x := inputs[0]
	op.Save(x)
	out, ok := x.Data().Transpose()
	if !ok {
		return nil, errors.New("cannot transpose tensor")
	}
	v := autograd.NewVariable(out, x.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *Transpose) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	out, ok := grad.Transpose()
	if !ok {
		return nil, errors.New("cannot transpose gradient")
	}
	return []tensor.Tensor{out}, nil
}
func (op *Tanh) Name() string {
	return "Tanh"
}
func (op *Tanh) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("tanh requires exactly one input")
	}
	x := inputs[0]
	op.Save(x)
	out := x.Data().Tanh()
	v := autograd.NewVariable(out, x.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *Tanh) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	output := op.Output().Data()
	out := output.TanhBackward(grad)
	return []tensor.Tensor{out}, nil
}
func (op *Sum) Name() string {
	return "Sum"
}
func (op *Sum) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("sum requires exactly one input")
	}
	x := inputs[0]
	op.Save(x)
	out := x.Data().SumTensor()
	v := autograd.NewVariable(out, x.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *Sum) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	input := op.Input(0).Data()
	out, ok := grad.Broadcast(input.Shape())
	if !ok {
		return nil, errors.New("sum backward broadcast failed")
	}
	return []tensor.Tensor{out}, nil
}
func (op *Sub) Name() string {
	return "Sub"
}
func (op *Sub) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 2 {
		return nil, errors.New("sub requires exactly 2 inputs")
	}
	a := inputs[0]
	b := inputs[1]
	op.Save(a, b)
	out := autograd.NewVariable(a.Data().Sub(b.Data()), a.RequiresGrad() || b.RequiresGrad())
	out.Node().Parents = []*autograd.Node{a.Node(), b.Node()}
	out.Node().Op = op
	op.SetOutput(out)
	return out, nil
}
func (op *Sub) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	return []tensor.Tensor{grad, grad.Neg()}, nil
}
func (op *Sigmoid) Name() string {
	return "Sigmoid"
}
func (op *Sigmoid) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("sigmoid requires exactly one input")
	}
	x := inputs[0]
	op.Save(x)
	out := x.Data().Sigmoid()
	v := autograd.NewVariable(out, x.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *Sigmoid) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	output := op.Output().Data()
	out := output.SigmoidBackward(grad)
	return []tensor.Tensor{out}, nil
}
func NewReshape(s shape.Shape) *Reshape {
	return &Reshape{newShape: s}
}
func (op *Reshape) Name() string {
	return "Reshape"
}
func (op *Reshape) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("reshape requires exactly one input")
	}
	x := inputs[0]
	op.Save(x)
	out, ok := x.Data().Reshape(op.newShape)
	if !ok {
		return nil, errors.New("invalid reshape")
	}
	v := autograd.NewVariable(out, x.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *Reshape) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	original := op.Input(0).Data().Shape()
	out, ok := grad.Reshape(original)
	if !ok {
		return nil, errors.New("reshape backward failed")
	}
	return []tensor.Tensor{out}, nil
}
func (op *ReLU) Name() string {
	return "ReLU"
}
func (op *ReLU) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("relu requires exactly one input")
	}
	x := inputs[0]
	op.Save(x)
	out := x.Data().ReLU()
	v := autograd.NewVariable(out, x.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *ReLU) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	input := op.Input(0).Data()
	mask := input.ReLUMask()
	out := grad.Mul(mask)
	return []tensor.Tensor{out}, nil
}
func (op *Neg) Name() string {
	return "Neg"
}
func (op *Neg) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("neg requires exactly 1 input")
	}
	x := inputs[0]
	op.Save(x)
	out := autograd.NewVariable(x.Data().Neg(), x.RequiresGrad())
	out.Node().Parents = []*autograd.Node{x.Node()}
	out.Node().Op = op
	op.SetOutput(out)
	return out, nil
}
func (op *Neg) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	return []tensor.Tensor{grad.Neg()}, nil
}
func (op *Mul) Name() string {
	return "Mul"
}
func (op *Mul) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 2 {
		return nil, errors.New("mul requires exactly 2 inputs")
	}
	a := inputs[0]
	b := inputs[1]
	op.Save(a, b)
	out := autograd.NewVariable(a.Data().Mul(b.Data()), a.RequiresGrad() || b.RequiresGrad())
	out.Node().Parents = []*autograd.Node{a.Node(), b.Node()}
	out.Node().Op = op
	op.SetOutput(out)
	return out, nil
}
func (op *Mul) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	return []tensor.Tensor{grad.Mul(op.Input(1).Data()), grad.Mul(op.Input(0).Data())}, nil
}
func (op *MSE) Name() string {
	return "MSELoss"
}
func (op *MSE) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 2 {
		return nil, errors.New("mse loss requires prediction and target")
	}
	prediction := inputs[0]
	target := inputs[1]
	op.Save(prediction, target)
	diff := prediction.Data().Sub(target.Data())
	squared := diff.Mul(diff)
	loss := squared.ReduceMean()
	out := autograd.NewVariable(loss, prediction.RequiresGrad())
	out.Node().Parents = []*autograd.Node{prediction.Node(), target.Node()}
	out.Node().Op = op
	op.SetOutput(out)
	return out, nil
}
func (op *MSE) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	pred := op.Input(0).Data()
	target := op.Input(1).Data()
	diff := pred.Sub(target)
	scale := float32(2.0 / float32(diff.NumElements()))
	dPred := diff.MulScalar(scale)
	dPred = dPred.ScalarMul(grad.FlatAt(0))
	dTarget := dPred.Neg()
	return []tensor.Tensor{dPred, dTarget}, nil
}
func (op *Mean) Name() string {
	return "Mean"
}
func (op *Mean) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("mean requires exactly one input")
	}
	x := inputs[0]
	op.Save(x)
	out := x.Data().ReduceMean()
	v := autograd.NewVariable(out, x.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *Mean) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	input := op.Input(0).Data()
	size := float32(input.NumElements())
	g := grad.DivScalar(size)
	out, ok := g.Broadcast(input.Shape())
	if !ok {
		return nil, errors.New("mean backward broadcast failed")
	}
	return []tensor.Tensor{out}, nil
}
func NewMaxPool2D(kernelH, kernelW, strideH, strideW int) *MaxPool2D {
	return &MaxPool2D{
		KernelH: kernelH,
		KernelW: kernelW,
		StrideH: strideH,
		StrideW: strideW,
	}
}
func (op *MaxPool2D) Name() string {
	return "MaxPool2D"
}
func (op *MaxPool2D) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("maxpool requires one input")
	}
	x := inputs[0]
	op.Save(x)
	dims := x.Data().Shape().Values()
	batch := dims[0]
	channels := dims[1]
	height := dims[2]
	width := dims[3]
	outH := (height-op.KernelH)/op.StrideH + 1
	outW := (width-op.KernelW)/op.StrideW + 1
	out := tensor.New(shape.New(batch, channels, outH, outW))
	count := batch * channels * outH * outW
	op.ArgMax = make([]int, count)
	for n := range batch {
		for c := range channels {
			for oh := range outH {
				for ow := range outW {
					maxValue := float32(math.Inf(-1))
					maxIndex := 0
					for kh := range op.KernelH {
						// for kh := 0; kh < op.KernelH; kh++ {
						// for kw := 0; kw < op.KernelW; kw++ {
						for kw := range op.KernelW {
							ih := oh*op.StrideH + kh
							iw := ow*op.StrideW + kw
							v := x.Data().At(n, c, ih, iw)
							if v > maxValue {
								maxValue = v
								maxIndex = ((n*channels+c)*height+ih)*width + iw
							}
						}
					}
					out.Set(maxValue, n, c, oh, ow)
					outputIndex := (((n*channels+c)*outH+oh)*outW + ow)
					op.ArgMax[outputIndex] = maxIndex
				}
			}
		}
	}
	v := autograd.NewVariable(out, x.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *MaxPool2D) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	input := op.Input(0).Data()
	dx := tensor.New(input.Shape())
	dims := grad.Shape().Values()
	batch := dims[0]
	channels := dims[1]
	outH := dims[2]
	outW := dims[3]
	index := 0
	for n := range batch {
		for c := range channels {
			for h := range outH {
				for w := range outW {
					inputIndex := op.ArgMax[index]
					old := dx.FlatAt(inputIndex)
					dx.FlatSet(inputIndex, old+grad.At(n, c, h, w))
					index++
				}
			}
		}
	}
	return []tensor.Tensor{dx}, nil
}
func (op *MatMul) Name() string {
	return "MatMul"
}
func (op *MatMul) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 2 {
		return nil, errors.New("matmul requires exactly 2 inputs")
	}
	a := inputs[0]
	b := inputs[1]
	op.Save(a, b)
	out, ok := a.Data().MatMul(b.Data())
	if !ok {
		return nil, errors.New("invalid matrix multiplication")
	}
	v := autograd.NewVariable(out, a.RequiresGrad() || b.RequiresGrad())
	v.Node().Parents = []*autograd.Node{a.Node(), b.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *MatMul) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	a := op.Input(0).Data()
	b := op.Input(1).Data()
	bt, ok := b.Transpose()
	if !ok {
		return nil, errors.New("cannot transpose rhs")
	}
	at, ok := a.Transpose()
	if !ok {
		return nil, errors.New("cannot transpose lhs")
	}
	da, ok := grad.MatMul(bt)
	if !ok {
		return nil, errors.New("cannot compute lhs gradient")
	}
	db, ok := at.MatMul(grad)
	if !ok {
		return nil, errors.New("cannot compute rhs gradient")
	}
	return []tensor.Tensor{da, db}, nil
}
func (op *Flatten) Name() string {
	return "Flatten"
}
func (op *Flatten) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 1 {
		return nil, errors.New("flatten requires exactly one input")
	}
	x := inputs[0]
	op.Save(x)
	out, ok := x.Data().Flatten()
	if !ok {
		return nil, errors.New("flatten failed")
	}
	v := autograd.NewVariable(out, x.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *Flatten) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	in := op.Input(0)
	out, ok := grad.Reshape(in.Data().Shape())
	if !ok {
		return nil, errors.New("reshape failed")
	}
	return []tensor.Tensor{out}, nil
}
func (op *Div) Name() string {
	return "Div"
}
func (op *Div) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 2 {
		return nil, errors.New("div requires exactly 2 inputs")
	}
	a := inputs[0]
	b := inputs[1]
	op.Save(a, b)
	out := autograd.NewVariable(a.Data().Div(b.Data()), a.RequiresGrad() || b.RequiresGrad())
	out.Node().Parents = []*autograd.Node{a.Node(), b.Node()}
	out.Node().Op = op
	op.SetOutput(out)
	return out, nil
}
func (op *Div) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	a := op.Input(0).Data()
	b := op.Input(1).Data()
	return []tensor.Tensor{grad.Div(b), grad.Mul(a).Div(b.Mul(b)).Neg()}, nil
}
func (op *CrossEntropyLoss) Name() string {
	return "CrossEntropyLoss"
}
func (op *CrossEntropyLoss) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 2 {
		return nil, errors.New("cross entropy requires prediction and target")
	}
	logits := inputs[0]
	target := inputs[1]
	op.Save(logits, target)
	logProb := logits.Data().LogSoftmaxDim(1)
	dims := logProb.Shape().Values()
	batch := dims[0]
	loss := float32(0)
	for i := range batch {
		class := int(target.Data().At(i))
		loss -= logProb.At(i, class)
	}
	loss /= float32(batch)
	outTensor := tensor.Scalar(loss)
	out := autograd.NewVariable(outTensor, logits.RequiresGrad())
	out.Node().Parents = []*autograd.Node{logits.Node()}
	out.Node().Op = op
	op.SetOutput(out)
	return out, nil
}
func (op *CrossEntropyLoss) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	logits := op.Input(0).Data()
	target := op.Input(1).Data()
	dims := logits.Shape().Values()
	batch := dims[0]
	classes := dims[1]
	prob := logits.SoftmaxDim(1)
	gradInput := tensor.New(logits.Shape())
	for i := range batch {
		class := int(target.At(i))
		for c := range classes {
			value := prob.At(i, c)
			if c == class {
				value -= 1
			}
			value /= float32(batch)
			gradInput.Set(value, i, c)
		}
	}
	gradValue := grad.FlatAt(0)
	gradInput = gradInput.ScalarMul(gradValue)
	return []tensor.Tensor{gradInput}, nil
}
func NewConv2D(strideH int, strideW int, paddingH int, paddingW int, kernelH int, kernelW int) *Conv2D {
	return &Conv2D{
		StrideH:  strideH,
		StrideW:  strideW,
		PaddingH: paddingH,
		PaddingW: paddingW,
		KernelH:  kernelH,
		KernelW:  kernelW,
	}
}
func (op *Conv2D) Name() string {
	return "Conv2D"
}
func (op *Conv2D) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 3 {
		return nil, errors.New("conv2d requires x weight bias")
	}
	x := inputs[0]
	w := inputs[1]
	b := inputs[2]
	op.Save(x, w, b)
	out := convForward(x.Data(), w.Data(), b.Data(), op)
	v := autograd.NewVariable(out, x.RequiresGrad() || w.RequiresGrad() || b.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node(), w.Node(), b.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *Conv2D) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	x := op.Input(0).Data()
	w := op.Input(1).Data()
	xShape := x.Shape().Values()
	wShape := w.Shape().Values()
	batch := xShape[0]
	inC := xShape[1]
	height := xShape[2]
	width := xShape[3]
	outC := wShape[0]
	dx := tensor.New(x.Shape())
	dw := tensor.New(w.Shape())
	db := tensor.New(shape.New(outC))
	outH := grad.Shape().Values()[2]
	outW := grad.Shape().Values()[3]
	for n := range batch {
		for oc := range outC {
			for oh := range outH {
				for ow := range outW {
					g := grad.At(n, oc, oh, ow)
					db.Set(db.At(oc)+g, oc)
					for ic := range inC {
						for kh := range op.KernelH {
							for kw := range op.KernelW {
								ih := oh*op.StrideH + kh - op.PaddingH
								iw := ow*op.StrideW + kw - op.PaddingW
								if ih < 0 || ih >= height || iw < 0 || iw >= width {
									continue
								}
								dw.Set(dw.At(oc, ic, kh, kw)+g*x.At(n, ic, ih, iw), oc, ic, kh, kw)
								dx.Set(dx.At(n, ic, ih, iw)+g*w.At(oc, ic, kh, kw), n, ic, ih, iw)
							}
						}
					}
				}
			}
		}
	}
	return []tensor.Tensor{dx, dw, db}, nil
}
func convForward(x tensor.Tensor, w tensor.Tensor, b tensor.Tensor, op *Conv2D) tensor.Tensor {
	dims := x.Shape().Values()
	batch := dims[0]
	inC := dims[1]
	height := dims[2]
	width := dims[3]
	wShape := w.Shape().Values()
	outC := wShape[0]
	outH := (height+2*op.PaddingH-op.KernelH)/op.StrideH + 1
	outW := (width+2*op.PaddingW-op.KernelW)/op.StrideW + 1
	out := tensor.New(shape.New(batch, outC, outH, outW))
	for n := range batch {
		for oc := range outC {
			for oh := range outH {
				for ow := range outW {
					sum := float32(0)
					for ic := range inC {
						for kh := range op.KernelH {
							for kw := range op.KernelW {
								ih := oh*op.StrideH + kh - op.PaddingH
								iw := ow*op.StrideW + kw - op.PaddingW
								if ih < 0 || ih >= height || iw < 0 || iw >= width {
									continue
								}
								sum += x.At(n, ic, ih, iw) * w.At(oc, ic, kh, kw)
							}
						}
					}
					sum += b.At(oc)
					out.Set(sum, n, oc, oh, ow)
				}
			}
		}
	}
	return out
}
func NewBatchNorm(channels int, eps float32) *BatchNorm {
	return &BatchNorm{
		Channels: channels,
		Eps:      eps,
	}
}
func (op *BatchNorm) Name() string {
	return "BatchNorm"
}
func (op *BatchNorm) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {
	if len(inputs) != 3 {
		return nil, errors.New("batchnorm requires input gamma beta")
	}
	x := inputs[0]
	gamma := inputs[1]
	beta := inputs[2]
	op.Save(x, gamma, beta)
	out := tensor.New(x.Data().Shape())
	shapev := x.Data().Shape().Values()
	batch := shapev[0]
	channels := shapev[1]
	height := shapev[2]
	width := shapev[3]
	mean := make([]float32, channels)
	variance := make([]float32, channels)
	size := float32(batch * height * width)
	for c := range channels {
		sum := float32(0)
		for n := range batch {
			for h := range height {
				for w := range width {
					sum += x.Data().At(n, c, h, w)
				}
			}
		}
		mean[c] = sum / size
	}
	for c := range channels {
		sum := float32(0)
		for n := range batch {
			for h := range height {
				for w := range width {
					v := x.Data().At(n, c, h, w) - mean[c]
					sum += v * v
				}
			}
		}
		variance[c] = sum / size
	}
	for n := range batch {
		for c := range channels {
			for h := range height {
				for w := range width {
					v := x.Data().At(n, c, h, w)
					norm := (v - mean[c]) / float32(math.Sqrt(float64(variance[c]+op.Eps)))
					y := norm*gamma.Data().At(c) + beta.Data().At(c)
					out.Set(y, n, c, h, w)
				}
			}
		}
	}
	op.Mean = tensor.New(shape.New(channels))
	op.Variance = tensor.New(shape.New(channels))
	v := autograd.NewVariable(out, x.RequiresGrad() || gamma.RequiresGrad() || beta.RequiresGrad())
	v.Node().Parents = []*autograd.Node{x.Node(), gamma.Node(), beta.Node()}
	v.Node().Op = op
	op.SetOutput(v)
	return v, nil
}
func (op *BatchNorm) Backward(grad tensor.Tensor) ([]tensor.Tensor, error) {
	x := op.Input(0).Data()
	gamma := op.Input(1).Data()
	dx := tensor.New(x.Shape())
	dgamma := tensor.New(gamma.Shape())
	dbeta := tensor.New(gamma.Shape())
	dims := x.Shape().Values()
	batch := dims[0]
	channels := dims[1]
	height := dims[2]
	width := dims[3]
	size := float32(batch * height * width)
	for c := range channels {
		sum := float32(0)
		for n := range batch {
			for h := range height {
				for w := range width {
					sum += grad.At(n, c, h, w)
				}
			}
		}
		dbeta.Set(sum, c)
	}
	for c := range channels {
		sum := float32(0)
		for n := range batch {
			for h := range height {
				for w := range width {
					v := x.At(n, c, h, w)
					norm := (v - op.Mean.At(c)) / float32(math.Sqrt(float64(op.Variance.At(c)+op.Eps)))
					sum += grad.At(n, c, h, w) * norm
				}
			}
		}
		dgamma.Set(sum, c)
	}
	for n := range batch {
		for c := range channels {
			sumDy := float32(0)
			sumDyNorm := float32(0)
			for i := range batch {
				for h := range height {
					for w := range width {
						g := grad.At(i, c, h, w)
						sumDy += g
						norm := (x.At(i, c, h, w) - op.Mean.At(c)) / float32(math.Sqrt(float64(op.Variance.At(c)+op.Eps)))
						sumDyNorm += g * norm
					}
				}
			}
			for h := range height {
				for w := range width {
					norm := (x.At(n, c, h, w) - op.Mean.At(c)) / float32(math.Sqrt(float64(op.Variance.At(c)+op.Eps)))
					d := grad.At(n, c, h, w)
					value := (gamma.At(c) / float32(math.Sqrt(float64(op.Variance.At(c)+op.Eps)))) * (size*d - sumDy - norm*sumDyNorm) / size
					dx.Set(value, n, c, h, w)
				}
			}
		}
	}
	return []tensor.Tensor{dx, dgamma, dbeta}, nil
}
func NewLeakyReLU(alpha float32) *LeakyReLU {
	return &LeakyReLU{NegativeSlope: alpha}
}
func (l *LeakyReLU) Forward(input *autograd.Variable) (*autograd.Variable, error) {
	out := input.Data().Clone()
	for i := 0; i < out.Len(); i++ {
		v := out.FlatAt(i)
		if v < 0 {
			v *= l.NegativeSlope
		}
		out.FlatSet(i, v)
	}
	return autograd.NewVariable(out, input.RequiresGrad()), nil
}
