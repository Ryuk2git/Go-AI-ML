package dataprep

import (
	"math"
	"strconv"

	"aiml/pkg/stats"
)

// ---------- Simple Imputation Methods ----------

// ImputeMean replaces missing numeric values with column mean.
func ImputeMean(col []string) []string {
	var nums []float64
	for _, v := range col {
		if v != "" && v != "NA" && v != "NaN" {
			if num, err := strconv.ParseFloat(v, 64); err == nil {
				nums = append(nums, num)
			}
		}
	}
	mean := stats.Mean(nums)

	for i, v := range col {
		if v == "" || v == "NA" || v == "NaN" {
			col[i] = strconv.FormatFloat(mean, 'f', 4, 64)
		}
	}
	return col
}

// ImputeMedian replaces missing numeric values with column median.
func ImputeMedian(col []string) []string {
	var nums []float64
	for _, v := range col {
		if v != "" && v != "NA" && v != "NaN" {
			if num, err := strconv.ParseFloat(v, 64); err == nil {
				nums = append(nums, num)
			}
		}
	}
	median := stats.Median(nums)

	for i, v := range col {
		if v == "" || v == "NA" || v == "NaN" {
			col[i] = strconv.FormatFloat(median, 'f', 4, 64)
		}
	}
	return col
}

// ImputeMode replaces missing numeric values with the mode.
func ImputeMode(col []string) []string {
	var nums []float64
	for _, v := range col {
		// Only parse valid numeric strings
		if v != "" && v != "NA" && v != "NaN" {
			if num, err := strconv.ParseFloat(v, 64); err == nil {
				nums = append(nums, num)
			}
		}
	}

	// Calculate the mode of the numeric values
	mode := stats.Mode(nums)

	// Replace missing values with the calculated mode
	for i, v := range col {
		if v == "" || v == "NA" || v == "NaN" {
			col[i] = strconv.FormatFloat(mode, 'f', 4, 64)
		}
	}

	return col
}

// ImputeConstant replaces missing values with a fixed constant.
func ImputeConstant(col []string, constant string) []string {
	for i, v := range col {
		if v == "" || v == "NA" || v == "NaN" {
			col[i] = constant
		}
	}
	return col
}

// ---------- Advanced Imputation Methods ----------

// ImputeKNN performs a simple KNN imputation for numeric values.
// For simplicity: average of k nearest neighbors (Euclidean distance on other features).
func ImputeKNN(data [][]string, targetCol int, k int) []string {
	rows := len(data)
	col := make([]string, rows)

	// Extract column
	for i := 0; i < rows; i++ {
		col[i] = data[i][targetCol]
	}

	// Find missing rows
	for i := 0; i < rows; i++ {
		if col[i] == "" || col[i] == "NA" || col[i] == "NaN" {
			// Compute distances to other rows
			var neighbors []float64
			for j := 0; j < rows; j++ {
				if i == j {
					continue
				}
				if val, err := strconv.ParseFloat(data[j][targetCol], 64); err == nil {
					dist := euclideanDistance(data[i], data[j], targetCol)
					if !math.IsNaN(dist) {
						neighbors = append(neighbors, val)
					}
				}
			}
			// Average neighbors
			if len(neighbors) > 0 {
				mean := stats.Mean(neighbors)
				col[i] = strconv.FormatFloat(mean, 'f', 4, 64)
			} else {
				col[i] = "0"
			}
		}
	}

	return col
}

// euclideanDistance computes distance between two rows (ignoring targetCol).
func euclideanDistance(a, b []string, targetCol int) float64 {
	var sum float64
	for i := range a {
		if i == targetCol {
			continue
		}
		x, err1 := strconv.ParseFloat(a[i], 64)
		y, err2 := strconv.ParseFloat(b[i], 64)
		if err1 == nil && err2 == nil {
			sum += math.Pow(x-y, 2)
		}
	}
	return math.Sqrt(sum)
}
