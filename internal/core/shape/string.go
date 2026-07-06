package shape

import "fmt"

func (s *Shape) String() string{

	return fmt.Sprint(s.dims)

}