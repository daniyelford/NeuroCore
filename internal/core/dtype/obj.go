/*
Package dtype defines all supported tensor element types.
*/
package dtype

type DType struct {
	name string
	size uint8
	kind Kind
}
