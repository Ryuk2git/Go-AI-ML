package model

import (
	"errors"
	"runtime"
	"sort"
	"sync"
)

// KNN supports classification when y is 0/1 labels.
type KNN struct {
	K int
	X [][]float64
	y []float64
}

// NewKNN creates and returns a new KNN model.
func NewKNN(k int) *KNN {
	return &KNN{K: k}
}

// Fit trains the model by simply storing the training data and labels.
// This is the "lazy" part of a KNN model.
func (m *KNN) Fit(X [][]float64, y []float64) error {
	if len(X) != len(y) {
		return errors.New("the number of feature vectors must match the number of labels")
	}
	m.X = X
	m.y = y
	return nil
}

// Predict finds the K-nearest neighbors for each test point and returns a prediction.
// This function is now parallelized using goroutines for better performance.
func (m *KNN) Predict(X [][]float64) []float64 {
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
		end := min(start+rowsPerWorker, len(X))
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				out[i] = m.predictSingle(X[i])
			}
		}(start, end)
	}

	wg.Wait()
	return out
}

// predictSingle finds the K-nearest neighbors for a single test point.
func (m *KNN) predictSingle(xi []float64) float64 {
	// A simple struct to hold a neighbor's squared distance and its label.
	type pair struct {
		d float64
		v float64
	}

	// We'll maintain a small sorted slice of the K-nearest neighbors found so far.
	nbrs := make([]pair, 0, m.K+1)

	// Iterate through all training data points.
	for j, xj := range m.X {
		distSquared := euclidSquared(xi, xj)
		neighbor := pair{d: distSquared, v: m.y[j]}

		// If the list of neighbors is not yet full, or if the current neighbor
		// is closer than the farthest one in the list, add it.
		if len(nbrs) < m.K {
			nbrs = append(nbrs, neighbor)
			sort.Slice(nbrs, func(a, b int) bool { return nbrs[a].d < nbrs[b].d })
		} else if distSquared < nbrs[len(nbrs)-1].d {
			nbrs[len(nbrs)-1] = neighbor
			sort.Slice(nbrs, func(a, b int) bool { return nbrs[a].d < nbrs[b].d })
		}
	}

	// Count the votes among the K-nearest neighbors.
	sum := 0.0
	for _, p := range nbrs {
		sum += p.v
	}

	// For binary labels (0/1), a simple majority vote determines the prediction.
	if sum/float64(len(nbrs)) >= 0.5 {
		return 1.0
	}
	return 0.0
}

// euclidSquared computes the squared Euclidean distance between two vectors.
// We use squared distance to avoid expensive square root operations during comparisons.
func euclidSquared(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
