package shape

// CanBroadcast reports whether s can broadcast to target.
//
// Full NumPy broadcasting rules will be implemented later.
func (s Shape) CanBroadcast(target Shape) bool {
	_ = target
	return false
}
