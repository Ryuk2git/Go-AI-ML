package stats

import (
	"math"
	"sort"
)

// Mean computes the average of a slice.
func Mean(x []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range x {
		sum += v
	}
	return sum / float64(n)
}

// Variance computes the variance of a slice in a single pass.
func Variance(x []float64) float64 {
	n := float64(len(x))
	if n == 0 {
		return 0
	}
	sum, sumSq := 0.0, 0.0
	for _, v := range x {
		sum += v
		sumSq += v * v
	}
	mean := sum / n
	return (sumSq / n) - (mean * mean)
}

// Std computes the standard deviation of a slice.
func Std(x []float64) float64 {
	return math.Sqrt(Variance(x))
}

// NormalizeInPlace normalizes the slice to zero mean and unit variance in a single pass.
func NormalizeInPlace(x []float64) {
	n := float64(len(x))
	if n == 0 {
		return
	}
	sum, sumSq := 0.0, 0.0
	for _, v := range x {
		sum += v
		sumSq += v * v
	}
	mean := sum / n
	variance := (sumSq / n) - (mean * mean)
	if variance == 0 {
		return
	}
	std := math.Sqrt(variance)
	for i := range x {
		x[i] = (x[i] - mean) / std
	}
}

// MinMax returns the minimum and maximum values in the slice.
func MinMax(x []float64) (float64, float64) {
	if len(x) == 0 {
		return 0, 0
	}
	min, max := x[0], x[0]
	for i := 1; i < len(x); i++ {
		if x[i] < min {
			min = x[i]
		} else if x[i] > max {
			max = x[i]
		}
	}
	return min, max
}

// Sum returns the sum of all elements in the slice.
func Sum(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += v
	}
	return s
}

// Median returns the median value of the slice (allocates a copy).
func Median(x []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	cp := make([]float64, n)
	copy(cp, x)
	sort.Float64s(cp)
	mid := n >> 1 // bitwise division by 2
	if n&1 == 0 { // even
		return (cp[mid-1] + cp[mid]) * 0.5
	}
	return cp[mid]
}

// MedianInPlace sorts and finds the median in-place (modifies input).
func MedianInPlace(x []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	sort.Float64s(x)
	mid := n >> 1
	if n&1 == 0 {
		return (x[mid-1] + x[mid]) * 0.5
	}
	return x[mid]
}

// Mode returns the most frequent value in the slice.
func Mode(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	counts := make(map[float64]int)
	maxCount := 0
	mode := x[0]
	for _, v := range x {
		counts[v]++
		if counts[v] > maxCount {
			maxCount = counts[v]
			mode = v
		}
	}
	return mode
}

// Percentile returns the p-th percentile value of the slice (0 <= p <= 100).
func Percentile(x []float64, p float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	min, max := MinMax(x)
	if p <= 0 {
		return min
	}
	if p >= 100 {
		return max
	}
	cp := make([]float64, n)
	copy(cp, x)
	sort.Float64s(cp)
	rank := p / 100 * float64(n-1)
	lower := int(rank)
	upper := lower + 1
	weight := rank - float64(lower)
	if upper >= n {
		return cp[lower]
	}
	return cp[lower]*(1-weight) + cp[upper]*weight
}

// PercentileInPlace returns the p-th percentile value of the slice (modifies input).
func PercentileInPlace(x []float64, p float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	min, max := MinMax(x)
	if p <= 0 {
		return min
	}
	if p >= 100 {
		return max
	}
	sort.Float64s(x)
	rank := p / 100 * float64(n-1)
	lower := int(rank)
	upper := lower + 1
	weight := rank - float64(lower)
	if upper >= n {
		return x[lower]
	}
	return x[lower]*(1-weight) + x[upper]*weight
}

// Covariance computes the covariance between two slices in a single pass.
func Covariance(x, y []float64) float64 {
	n := float64(len(x))
	if n == 0 || len(y) != len(x) {
		return 0
	}
	sumX, sumY, sumXY := 0.0, 0.0, 0.0
	for i := range x {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
	}
	meanX := sumX / n
	meanY := sumY / n
	return (sumXY / n) - (meanX * meanY)
}

// Correlation computes the Pearson correlation coefficient between two slices in a single pass.
func Correlation(x, y []float64) float64 {
	n := float64(len(x))
	if n == 0 || len(y) != len(x) {
		return 0
	}
	var sumX, sumY, sumXY, sumX2, sumY2 float64
	for i := range x {
		xi, yi := x[i], y[i]
		sumX += xi
		sumY += yi
		sumXY += xi * yi
		sumX2 += xi * xi
		sumY2 += yi * yi
	}
	numerator := n*sumXY - sumX*sumY
	denominator := math.Sqrt((n*sumX2 - sumX*sumX) * (n*sumY2 - sumY*sumY))
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}
