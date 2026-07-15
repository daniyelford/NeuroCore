package runtime

import "testing"

func TestNewRuntime(t *testing.T) {

	rt := New()

	if rt == nil {
		t.Fatal("runtime is nil")
	}

}