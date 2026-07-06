package stride

import "errors"

var (
	ErrEmptyStride = errors.New("stride cannot be empty")
	ErrInvalidStride = errors.New("invalid stride")
	ErrIndexOutOfRange = errors.New("index out of range")
)