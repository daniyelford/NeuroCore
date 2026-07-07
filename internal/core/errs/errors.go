package errs

import "errors"

var (

	ErrInvalidArgument = errors.New("invalid argument")

	ErrOutOfRange = errors.New("index out of range")

	ErrEmpty = errors.New("empty value")

	ErrNil = errors.New("nil value")

	ErrNotImplemented = errors.New("not implemented")

)