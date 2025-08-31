package NeuralNetwork

import "math"

// Mean Squared Error(MSE) and its gradient for regression
// Use this loss when predicting continuous values (regression problems)
func MSE(yTrue, yPred []float64) (float64, []float64) {
	n := len(yTrue)
	s := 0.0
	grad := make([]float64, n)

	for i := range n {
		e := yPred[i] - yTrue[i]
		s += e * e
		grad[i] = 2 * e / float64(n)
	}
	return s / float64(n), grad
}

// Binary cross-entropy loss and gradient for logistic regression
// Use this loss when predicting probabilities for two classes (binary classification)
func BCE(yTrue, yPred []float64) (float64, []float64) {
	n := len(yTrue)
	s := 0.0
	grad := make([]float64, n)

	for i := range n {
		p := math.Min(math.Max(yPred[i], 1e-12), 1-1e-12)
		y := yTrue[i]
		s += -(y*math.Log(p) + (1-y)*math.Log(1-p))
		grad[i] = (p - y) / float64(n)
	}
	return s / float64(n), grad
}
