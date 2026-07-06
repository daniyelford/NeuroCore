package shape

import "errors"

var (

	ErrEmptyShape=

		errors.New("shape cannot be empty")

	ErrInvalidDimension=

		errors.New("invalid dimension")

)