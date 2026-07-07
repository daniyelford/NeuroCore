package impl

func (t *TensorImpl) ShallowClone() *TensorImpl {

	t.storage.Acquire()

	x := *t

	return &x

}
func (t *TensorImpl) Clone() *TensorImpl {

	buf := t.storage.Buffer().Clone()

	st := storage.New(buf)

	x := *t

	x.storage = st

	x.version.Store(1)

	return &x

}