package dataprep

import "math"

// PolynomialFeatures generates polynomial features up to the given degree.
func PolynomialFeatures(X [][]float64, degree int) [][]float64 {
	// For simplicity, only implements degree 2 (quadratic) for now.
	rows, cols := len(X), len(X[0])
	out := make([][]float64, rows)
	for i := range rows {
		features := make([]float64, cols+cols*(cols+1)/2)
		copy(features, X[i])
		idx := cols
		for j := 0; j < cols; j++ {
			for k := j; k < cols; k++ {
				features[idx] = X[i][j] * X[i][k]
				idx++
			}
		}
		out[i] = features
	}
	return out
}

// BinContinuous bins continuous values into n bins.
func BinContinuous(X []float64, nBins int) []int {
	min, max := X[0], X[0]
	for _, v := range X {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	bins := make([]int, len(X))
	width := (max - min) / float64(nBins)
	for i, v := range X {
		b := int((v - min) / width)
		if b >= nBins {
			b = nBins - 1
		}
		bins[i] = b
	}
	return bins
}

// LogTransform applies log(x+1) to each value.
func LogTransform(X []float64) []float64 {
	out := make([]float64, len(X))
	for i, v := range X {
		out[i] = math.Log1p(v)
	}
	return out
}

// FeatureSelect selects columns by indices.
func FeatureSelect(X [][]float64, indices []int) [][]float64 {
	out := make([][]float64, len(X))
	for i, row := range X {
		selected := make([]float64, len(indices))
		for j, idx := range indices {
			selected[j] = row[idx]
		}
		out[i] = selected
	}
	return out
}
