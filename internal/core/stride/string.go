package stride

import "fmt"

func (s *Stride) String() string {
	return fmt.Sprint(s.values)
}