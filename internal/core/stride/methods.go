package stride

func (s *Stride) Rank() int {
	return len(s.values)
}

func (s *Stride) Values() []int {

	cp := make([]int, len(s.values))

	copy(cp, s.values)

	return cp
}

func (s *Stride) At(i int) (int, bool) {

	if i < 0 || i >= len(s.values) {
		return 0, false
	}

	return s.values[i], true
}

func (s *Stride) Last() int {
	return s.values[len(s.values)-1]
}

func FromShape(shape []int) (*Stride,error){

	if len(shape)==0{
		return nil,ErrEmptyStride
	}

	values:=make([]int,len(shape))

	step:=1

	for i:=len(shape)-1;i>=0;i--{

		values[i]=step

		step*=shape[i]

	}

	return New(values...)

}

func (s *Stride) Equal(other *Stride) bool{

	if len(s.values)!=len(other.values){
		return false
	}

	for i:=range s.values{

		if s.values[i]!=other.values[i]{
			return false
		}

	}

	return true

}