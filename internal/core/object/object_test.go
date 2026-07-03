package object

import "testing"

func TestObject(t *testing.T){

	obj:=New("tensor")

	if obj.Name()!="tensor"{

		t.Fatal()

	}

}