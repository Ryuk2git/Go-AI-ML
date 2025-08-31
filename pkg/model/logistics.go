package model

import (
	"errors"
	"math/rand"
	"runtime"
	"sync"

	"aiml/pkg/NeuralNetwork"
	"aiml/pkg/optim"
)

// LogisticRegression (binary) with sigmoid.
// This struct holds the model parameters and hyperparameters for training.
type LogisticRegression struct {
	W         []float64 // weights
	b         float64   // bias
	Lr        float64
	Epochs    int
	BatchSize int
}

// NewLogisticRegression initializes a new Logistic Regression model.
// It sets the initial weights and stores the hyperparameters for later use.
func NewLogisticRegression(nFeatures int, lr float64, epochs int, batchSize int) *LogisticRegression {
	w := make([]float64, nFeatures)
	// Initialize weights with small random values to break symmetry.
	for i := range w {
		w[i] = rand.NormFloat64() * 0.01
	}
	return &LogisticRegression{
		W:         w,
		b:         0.0,
		Lr:        lr,
		Epochs:    epochs,
		BatchSize: batchSize,
	}
}

// PredictProba returns the probability scores (between 0 and 1) for each input row in X.
// It uses goroutines to parallelize the prediction process for efficiency.
func (m *LogisticRegression) PredictProba(X [][]float64) []float64 {
	if len(X) == 0 {
		return nil
	}
	out := make([]float64, len(X))
	var wg sync.WaitGroup

	// Determine the number of workers based on available CPU cores.
	workers := runtime.GOMAXPROCS(0)
	rowsPerWorker := (len(X) + workers - 1) / workers

	for w := 0; w < workers; w++ {
		start := w * rowsPerWorker
		end := start + rowsPerWorker
		if end > len(X) {
			end = len(X)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				row := X[i]
				sum := m.b
				for j, v := range row {
					sum += m.W[j] * v
				}
				out[i] = NeuralNetwork.Sigmoid(sum)
			}
		}(start, end)
	}
	wg.Wait()
	return out
}

// Predict returns the class labels (0 or 1) based on a 0.5 probability threshold.
func (m *LogisticRegression) Predict(X [][]float64) []float64 {
	proba := m.PredictProba(X)
	out := make([]float64, len(proba))
	for i, p := range proba {
		if p >= 0.5 {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
	return out
}

// Fit trains the model using mini-batch gradient descent.
// It consumes data from a channel and updates the model's weights and bias.
func (m *LogisticRegression) Fit(batches <-chan struct {
	X [][]float64
	Y []float64
}) error {
	// Initialize the SGD optimizer using the learning rate from the struct.
	opt := optim.NewSGD(m.Lr)

	for ep := 0; ep < m.Epochs; ep++ {
		// Iterate over batches received from the channel.
		for batch := range batches {
			// Validate that the number of features matches the model's weight count.
			if len(m.W) != len(batch.X[0]) {
				return errors.New("feature count mismatch between model and batch data")
			}

			// Forward Pass: Get the predicted probabilities.
			p := m.PredictProba(batch.X)

			// Calculate the derivative of the Binary Cross-Entropy (BCE) loss with respect to the predictions.
			_, dy := NeuralNetwork.BCE(batch.Y, p)

			// Initialize slices for the gradients of weights and bias.
			gW := make([]float64, len(m.W))
			gb := 0.0

			// Backward Pass: Calculate the gradients of the loss with respect to the weights (gW) and bias (gb).
			for i, row := range batch.X {
				d := dy[i]
				for j, xij := range row {
					gW[j] += d * xij
				}
				gb += d
			}

			// Use the SGD optimizer to perform the weight updates.
			opt.Step(m.W, gW)

			// Update the bias separately.
			m.b -= m.Lr * gb
		}
	}
	return nil
}
