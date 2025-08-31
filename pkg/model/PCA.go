package model

import (
	"errors"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// PCA via power iteration for top-k components.
type PCA struct {
	K          int
	MaxIters   int
	Means      []float64
	Components [][]float64 // K x p, each a unit vector
	Explained  []float64   // approx eigenvalues
}

// NewPCA creates and returns a new PCA model.
func NewPCA(k int, maxIters int) *PCA {
	return &PCA{K: k, MaxIters: maxIters}
}

// Fit trains the PCA model by computing the top K principal components.
// The algorithm uses a parallelized power iteration with deflation for efficiency.
func (pca *PCA) Fit(X [][]float64) error {
	if len(X) == 0 {
		return errors.New("input data cannot be empty")
	}

	n, d := len(X), len(X[0])

	// --- Step 1: Parallel Data Centering ---
	pca.Means = make([]float64, d)
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)

	// Calculate sums for each feature in parallel.
	sums := make([][]float64, workers)
	for i := range sums {
		sums[i] = make([]float64, d)
	}

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
		go func(start, end int, workerID int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < d; j++ {
					sums[workerID][j] += X[i][j]
				}
			}
		}(start, end, w)
	}
	wg.Wait()

	// Consolidate the sums and compute the means.
	for j := 0; j < d; j++ {
		for w := 0; w < workers; w++ {
			pca.Means[j] += sums[w][j]
		}
		pca.Means[j] /= float64(n)
	}

	// Create the centered matrix Z.
	Z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z := make([]float64, d)
		for j := 0; j < d; j++ {
			z[j] = X[i][j] - pca.Means[j]
		}
		Z[i] = z
	}

	// --- Step 2: Deflation Method (Find one component at a time) ---
	pca.Components = make([][]float64, 0, pca.K)
	pca.Explained = make([]float64, 0, pca.K)

	for comp := 0; comp < pca.K; comp++ {
		// Initialize a random unit vector for power iteration.
		v := make([]float64, d)
		for j := 0; j < d; j++ {
			v[j] = rand.Float64()
		}
		v = normalize(v)

		// Power iteration to find the eigenvector.
		for t := 0; t < pca.MaxIters; t++ {
			// w = (Z^T (Z v))
			Zv := make([]float64, n)
			w := make([]float64, d)

			// Parallel matrix-vector multiplication: Z * v
			wg.Add(workers)
			for wID := 0; wID < workers; wID++ {
				go func(id int) {
					defer wg.Done()
					start := id * rowsPerWorker
					end := start + rowsPerWorker
					if end > n {
						end = n
					}
					for i := start; i < end; i++ {
						s := 0.0
						for j := 0; j < d; j++ {
							s += Z[i][j] * v[j]
						}
						Zv[i] = s
					}
				}(wID)
			}
			wg.Wait()

			// Parallel matrix-vector multiplication: Z^T * Zv
			colsPerWorker := (d + workers - 1) / workers
			wg.Add(workers)
			for wID := 0; wID < workers; wID++ {
				go func(id int) {
					defer wg.Done()
					startCol := id * colsPerWorker
					endCol := startCol + colsPerWorker
					if endCol > d {
						endCol = d
					}
					for j := startCol; j < endCol; j++ {
						s := 0.0
						for i := 0; i < n; i++ {
							s += Z[i][j] * Zv[i]
						}
						w[j] = s
					}
				}(wID)
			}
			wg.Wait()

			v = normalize(w)
		}

		// Calculate explained variance (approximate eigenvalue).
		lam := 0.0
		for i := 0; i < n; i++ {
			s := 0.0
			for j := 0; j < d; j++ {
				s += Z[i][j] * v[j]
			}
			lam += s * s
		}
		lam /= float64(n - 1)
		pca.Explained = append(pca.Explained, lam)
		pca.Components = append(pca.Components, v)

		// --- Step 3: Parallel Deflation of the Data Matrix ---
		// Z = Z - (Z*v) * v^T
		// Subtract the effect of the component we just found.
		wg.Add(workers)
		for wID := 0; wID < workers; wID++ {
			go func(id int) {
				defer wg.Done()
				startRow := id * rowsPerWorker
				endRow := startRow + rowsPerWorker
				if endRow > n {
					endRow = n
				}
				for i := startRow; i < endRow; i++ {
					vecProduct := 0.0
					for j := 0; j < d; j++ {
						vecProduct += Z[i][j] * v[j]
					}
					for j := 0; j < d; j++ {
						Z[i][j] -= vecProduct * v[j]
					}
				}
			}(wID)
		}
		wg.Wait()
	}

	return nil
}

// Transform projects the input data onto the principal components.
func (pca *PCA) Transform(X [][]float64) ([][]float64, error) {
	if len(X) == 0 {
		return nil, errors.New("input data cannot be empty")
	}

	n, d := len(X), len(X[0])
	if d != len(pca.Means) {
		return nil, errors.New("feature count mismatch between input and training data")
	}

	// Center the new data.
	Z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z := make([]float64, d)
		for j := 0; j < d; j++ {
			z[j] = X[i][j] - pca.Means[j]
		}
		Z[i] = z
	}

	// Project the centered data onto the principal components.
	components := pca.Components
	transformed := make([][]float64, n)

	// Parallelize the transformation process.
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
				t := make([]float64, pca.K)
				for k := 0; k < pca.K; k++ {
					s := 0.0
					for j := 0; j < d; j++ {
						s += Z[i][j] * components[k][j]
					}
					t[k] = s
				}
				transformed[i] = t
			}
		}(start, end)
	}
	wg.Wait()

	return transformed, nil
}

// normalize normalizes a vector to have unit length.
func normalize(v []float64) []float64 {
	sumSquared := 0.0
	for _, val := range v {
		sumSquared += val * val
	}
	norm := math.Sqrt(sumSquared)

	if norm == 0 {
		return v
	}

	out := make([]float64, len(v))
	for i, val := range v {
		out[i] = val / norm
	}
	return out
}
