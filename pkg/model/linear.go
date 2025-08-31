package model

import (
	"errors"
	"math/rand"
	"runtime"
	"sync"

	"aiml/pkg/NeuralNetwork"
	"aiml/pkg/data"
	"aiml/pkg/optim"
)

// LinearRegression via mini-batch gradient descent. Robust & scalable.
type LinearRegression struct {
	W         []float64 // weights
	b         float64   // bias
	Lr        float64
	Epochs    int
	BatchSize int
}

// NewLinearRegression initializes a new Linear Regression model with the specified parameters.
func NewLinearRegression(nFeatures int, lr float64, epochs int, batchSize int) *LinearRegression {
	w := make([]float64, nFeatures)
	// No need to call rand.Seed since Go 1.20, as the global generator is automatically seeded.
	for i := range w {
		w[i] = rand.NormFloat64() * 0.01
	}
	return &LinearRegression{W: w, b: 0, Lr: lr, Epochs: epochs, BatchSize: batchSize}
}

// Predict returns predictions for rows in X (rows of features).
// This function uses goroutines and a WaitGroup to parallelize predictions across CPU cores,
// which is highly efficient for large datasets.
func (m *LinearRegression) Predict(X [][]float64) []float64 {
	if len(X) == 0 {
		return nil
	}
	pred := make([]float64, len(X))
	var wg sync.WaitGroup

	workers := runtime.GOMAXPROCS(0)
	rowsPerWorker := (len(X) + workers - 1) / workers

	for w := 0; w < workers; w++ {
		s := w * rowsPerWorker
		e := s + rowsPerWorker
		if e > len(X) {
			e = len(X)
		}
		if s >= e {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				row := X[i]
				sum := m.b
				for j, v := range row {
					sum += m.W[j] * v
				}
				pred[i] = sum
			}
		}(s, e)
	}
	wg.Wait()
	return pred
}

// Fit trains the model via mini-batch SGD using a channel to receive batches.
// It uses the Lr and Epochs values stored in the struct.
func (m *LinearRegression) Fit(batches <-chan data.Batch, epochs int, optimizer *optim.SGD) error {
	// Initialize the SGD optimizer using the learning rate from the struct.
	opt := optim.NewSGD(m.Lr)

	for ep := 0; ep < m.Epochs; ep++ {
		for batch := range batches {
			if len(m.W) != len(batch.X[0]) {
				return errors.New("feature count mismatch between model and batch data")
			}
			yhat := m.Predict(batch.X)
			_, dy := NeuralNetwork.MSE(batch.Y, yhat)
			gW := make([]float64, len(m.W))
			gb := 0.0
			for i, row := range batch.X {
				d := dy[i]
				for j, xij := range row {
					gW[j] += d * xij
				}
				gb += d
			}

			// Correctly use the optimizer to update the weights.
			opt.Step(m.W, gW)

			// The bias is a single scalar and is updated separately.
			m.b -= m.Lr * gb
		}
	}
	return nil
}

// Bias returns the current bias value of the model.
func (m *LinearRegression) Bias() float64 {
	return m.b
}
