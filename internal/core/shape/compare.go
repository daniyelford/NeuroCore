package shape

func (s *Shape) Equal(other *Shape) bool {

	if len(s.dims)!=len(other.dims){

		return false

	}

	for i:=range s.dims{

		if s.dims[i]!=other.dims[i]{

			return false

		}

	}

	return true

}