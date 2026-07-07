package impl

func (t *TensorImpl) Reshape(
	sh *shape.Shape,
	str *stride.Stride,
) *TensorImpl {

	x := t.ShallowClone()

	x.shape = sh

	x.stride = str

	return x

}