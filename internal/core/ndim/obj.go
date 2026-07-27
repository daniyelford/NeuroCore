/*
Package ndim provides common N-dimensional vector primitives.

It is shared by Shape and Stride.
*/
package ndim

// Vector represents an immutable N-dimensional vector.
type Vector struct {
	values []int
}
