package ndim

import "fmt"

func (v Vector) String() string {
	return fmt.Sprint(v.values)
}
