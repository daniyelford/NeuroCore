package stride

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
)

func TestFromShape(t *testing.T) {

	s := shape.New(3, 224, 224)

	st := FromShape(s)

	if !st.Equal(New(50176, 224, 1)) {
		t.Fatal("invalid stride")
	}

}

func TestOffset(t *testing.T) {

	st := New(50176, 224, 1)

	offset := st.Offset(2, 10, 15)

	if offset != 102687 {
		t.Fatal()
	}

}
func TestCompute(t *testing.T) {

	sh := shape.New(3, 224, 224)

	st := Compute(sh)

	if !st.Equal(New(50176, 224, 1)) {
		t.Fatal()
	}

}
func TestClone(t *testing.T) {

	s := New(6, 2, 1)

	c := s.Clone()

	if !c.Equal(s) {
		t.Fatal()
	}

}
func TestLast(t *testing.T) {

	s := New(4, 2, 1)

	if s.Last() != 1 {
		t.Fatal()
	}

}
