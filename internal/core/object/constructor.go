package object

func New(name string)*Object{

	return &Object{

		id:NewID(),

		name:name,

		metadata:make(map[string]any),

	}

}