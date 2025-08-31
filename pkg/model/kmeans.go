package model

import (
	"errors"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// KMeans is an unsupervised learning model that partitions data points into K clusters.
type KMeans struct {
	K         int
	MaxIter   int
	Centroids [][]float64
	Inertia   float64 // Sum of squared distances to nearest centroid
}

// NewKMeans creates and returns a new KMeans model with specified K and max iterations.
func NewKMeans(k int, maxIter int) *KMeans {
	return &KMeans{
		K:       k,
		MaxIter: maxIter,
	}
}

// Fit trains the KMeans model by iteratively finding centroids.
// It returns an error if the input data is invalid.
func (m *KMeans) Fit(X [][]float64) error {
	if len(X) == 0 {
		return errors.New("input data cannot be empty")
	}

	n, p := len(X), len(X[0])
	if n < m.K {
		return errors.New("number of data points is less than K")
	}

	// Initialize centroids by picking random distinct points from the data.
	// This is a common initialization strategy (K-means++ is an alternative).
	// idxs := rand.Perm(n)
	idxs := m.initCenters(X)
	m.Centroids = make([][]float64, m.K)
	for i := 0; i < m.K; i++ {
		c := make([]float64, p)
		copy(c, X[idxs[i]])
		m.Centroids[i] = c
	}

	assign := make([]int, n)
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)

	// Main K-Means loop
	for it := 0; it < m.MaxIter; it++ {
		changed := false
		m.Inertia = 0.0

		// === Parallel Assignment Step ===
		// Assign each data point to the nearest centroid.
		// Divide the work into chunks for parallel processing.
		rowsPerWorker := (n + workers - 1) / workers
		for w := 0; w < workers; w++ {
			start := w * rowsPerWorker
			end := start + rowsPerWorker
			if end > n {
				end = n
			}
			if start >= end {
				continue
			}

			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				for i := start; i < end; i++ {
					best, bestdSquared := -1, math.MaxFloat64
					for k := 0; k < m.K; k++ {
						dSquared := euclidSquared(X[i], m.Centroids[k])
						if dSquared < bestdSquared {
							bestdSquared = dSquared
							best = k
						}
					}
					// Only update if the assignment has changed.
					if assign[i] != best {
						changed = true
					}
					assign[i] = best
				}
			}(start, end)
		}
		wg.Wait()

		// === Update Step ===
		// Calculate the new centroids based on the mean of the assigned points.
		sums := make([][]float64, m.K)
		counts := make([]int, m.K)
		for k := 0; k < m.K; k++ {
			sums[k] = make([]float64, p)
		}
		for i := 0; i < n; i++ {
			k := assign[i]
			counts[k]++
			for j := 0; j < p; j++ {
				sums[k][j] += X[i][j]
			}
			// Update Inertia (sum of squared distances)
			m.Inertia += euclidSquared(X[i], m.Centroids[k])
		}

		for k := 0; k < m.K; k++ {
			if counts[k] == 0 {
				continue // Skip if a cluster is empty
			}
			for j := 0; j < p; j++ {
				m.Centroids[k][j] = sums[k][j] / float64(counts[k])
			}
		}

		// If no assignments changed, the algorithm has converged.
		if !changed {
			break
		}
	}
	return nil
}

// Predict assigns each data point to its nearest centroid and returns the cluster assignments.
// This function is parallelized for efficiency.
func (m *KMeans) Predict(X [][]float64) ([]int, error) {
	if len(X) == 0 {
		return nil, errors.New("input data for prediction cannot be empty")
	}

	n, p := len(X), len(X[0])
	if p != len(m.Centroids[0]) {
		return nil, errors.New("feature count mismatch between input data and model centroids")
	}

	assignments := make([]int, n)
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)
	rowsPerWorker := (n + workers - 1) / workers

	for w := 0; w < workers; w++ {
		start := w * rowsPerWorker
		end := start + rowsPerWorker
		if end > n {
			end = n
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				best, bestdSquared := -1, math.MaxFloat64
				for k := 0; k < m.K; k++ {
					dSquared := euclidSquared(X[i], m.Centroids[k])
					if dSquared < bestdSquared {
						bestdSquared = dSquared
						best = k
					}
				}
				assignments[i] = best
			}
		}(start, end)
	}
	wg.Wait()

	return assignments, nil
}

func (m *KMeans) initCenters(X [][]float64) []int {
	n, d := len(X), len(X[0])
	m.Centroids = make([][]float64, m.K)
	idxs := make([]int, m.K)

	// First center: pick randomly
	idx := rand.Intn(n)
	m.Centroids[0] = append([]float64{}, X[idx]...)
	idxs[0] = idx

	// Remaining centers
	for k := 1; k < m.K; k++ {
		distSq := make([]float64, n)
		total := 0.0
		for i, x := range X {
			minDist := math.MaxFloat64
			for _, c := range m.Centroids[:k] {
				d2 := 0.0
				for j := 0; j < d; j++ {
					dx := x[j] - c[j]
					d2 += dx * dx
				}
				if d2 < minDist {
					minDist = d2
				}
			}
			distSq[i] = minDist
			total += minDist
		}

		r := rand.Float64() * total
		cumulative := 0.0
		for i, d2 := range distSq {
			cumulative += d2
			if cumulative >= r {
				m.Centroids[k] = append([]float64{}, X[i]...)
				idxs[k] = i
				break
			}
		}
	}

	return idxs
}
