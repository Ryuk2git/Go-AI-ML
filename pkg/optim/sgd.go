package optim

// Stochastic Gradient Descent optimizer withg learning rate
type SGD struct{ LearningRate float64 }

func NewSGD(lr float64) *SGD { return &SGD{LearningRate: lr} }

func (o *SGD) Step(weights, grads []float64) { // in-place update using pointer receiver
	for i := range weights {
		weights[i] -= o.LearningRate * grads[i]
	}
}
