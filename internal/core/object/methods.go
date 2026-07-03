package object

func (o *Object) ID() uint64 {
	return o.id
}

func (o *Object) Name() string {
	return o.name
}

func (o *Object) SetName(name string) {
	o.name = name
}