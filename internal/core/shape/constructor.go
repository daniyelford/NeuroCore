func New(dims ...int) (*Shape,error){

	if len(dims)==0{
		return nil,ErrEmptyShape
	}

	size:=1

	cp:=make([]int,len(dims))

	for i,d:=range dims{

		if d<=0{
			return nil,ErrInvalidDimension
		}

		cp[i]=d

		size*=d

	}

	return &Shape{

		dims:cp,

		size:size,

	},nil

}