package NeuralNetwork

import "math"

func Sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

func SigmoidPrime(x float64) float64 { s := Sigmoid(x); return s * (1 - s) }

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
