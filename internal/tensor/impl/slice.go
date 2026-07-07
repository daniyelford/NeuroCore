package impl

func (t *TensorImpl) Slice(offset int) *TensorImpl {

	x := t.ShallowClone()

	x.offset += offset

	return x

}
