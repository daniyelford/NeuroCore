package errors

import "errors"

var (
	ErrEmptyData = errors.New("empty data")

	ErrShape = errors.New("invalid shape")

	ErrDevice = errors.New("invalid device")
)