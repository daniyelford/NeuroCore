package autograd

var gradEnabled = true

func EnableGrad() {

	gradEnabled = true

}

func DisableGrad() {

	gradEnabled = false

}

func GradEnabled() bool {

	return gradEnabled

}
