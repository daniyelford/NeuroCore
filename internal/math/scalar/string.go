package scalar

import "fmt"

func (s *Scalar[T]) String() string {

	return fmt.Sprintf("%v", s.value)

}