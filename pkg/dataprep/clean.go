package dataprep

import (
	"fmt"
	"math"
	"strconv"

	"aiml/pkg/stats"
)

// HandleMissingValues dynamically selects an imputation strategy
// based on data type, distribution, and missing value ratio.
// data: 2D slice representing dataset (rows x columns)
// threshold: if %missing > threshold, drop the column
func HandleMissingValues(data [][]string, threshold float64) [][]string {
	if len(data) == 0 {
		return data
	}

	rows := len(data)
	cols := len(data[0])

	// Track columns to keep
	var cleaned [][]string
	keepColumns := make([]bool, cols)

	// Iterate columns
	for c := range cols {
		missingCount := 0
		colValues := []string{}

		for r := range rows {
			val := data[r][c]
			if val == "" || val == "NA" || val == "NaN" {
				missingCount++
			}
			colValues = append(colValues, val)
		}

		missingRatio := float64(missingCount) / float64(rows)

		if missingRatio > threshold {
			fmt.Printf("Dropping column %d (%.2f%% missing)\n", c, missingRatio*100)
			keepColumns[c] = false
			continue
		}

		// Detect numeric vs categorical
		isNumeric := true
		var numericVals []float64
		for _, v := range colValues {
			if v == "" || v == "NA" || v == "NaN" {
				continue
			}
			num, err := strconv.ParseFloat(v, 64)
			if err != nil {
				isNumeric = false
				break
			}
			numericVals = append(numericVals, num)
		}

		// --- Strategy Selection ---
		if isNumeric {
			// Check skewness (simple heuristic: mean vs median distance)
			mean := stats.Mean(numericVals)
			median := stats.Median(numericVals)
			skew := math.Abs(mean-median) / (stats.Std(numericVals) + 1e-9)

			if missingRatio < 0.05 {
				// Low missingness → mean
				colValues = ImputeMean(colValues)
				fmt.Printf("Column %d: Imputed with MEAN\n", c)
			} else if skew > 1.0 {
				// Skewed → median
				colValues = ImputeMedian(colValues)
				fmt.Printf("Column %d: Imputed with MEDIAN (skewed)\n", c)
			} else if missingRatio < 0.2 {
				// Moderate missingness → KNN
				colValues = ImputeKNN(data, c, 3)
				fmt.Printf("Column %d: Imputed with KNN\n", c)
			} else {
				// High missingness but kept → constant
				colValues = ImputeConstant(colValues, "0")
				fmt.Printf("Column %d: Imputed with CONSTANT\n", c)
			}

		} else {
			// Categorical
			if missingRatio < 0.1 {
				colValues = ImputeMode(colValues)
				fmt.Printf("Column %d: Imputed with MODE\n", c)
			} else {
				colValues = ImputeConstant(colValues, "Unknown")
				fmt.Printf("Column %d: Imputed with CONSTANT (Unknown)\n", c)
			}
		}

		// Mark as keep
		keepColumns[c] = true

		// Update column back into dataset
		for r := range rows {
			data[r][c] = colValues[r]
		}
	}

	// Rebuild dataset with kept columns
	for r := range rows {
		var row []string
		for c := range cols {
			if keepColumns[c] {
				row = append(row, data[r][c])
			}
		}
		cleaned = append(cleaned, row)
	}

	return cleaned
}

// DropDuplicates removes duplicate rows from the dataset.
func DropDuplicates(X [][]float64) [][]float64 {
	seen := make(map[string]struct{})
	out := [][]float64{}
	for _, row := range X {
		key := fmt.Sprint(row)
		if _, ok := seen[key]; !ok {
			seen[key] = struct{}{}
			out = append(out, row)
		}
	}
	return out
}

// PolynomialFeaturesWithNames expands features with polynomial combinations and generates proper names
func PolynomialFeaturesWithNames(X [][]float64, headers []string, degree int) ([][]float64, []string) {
	nSamples := len(X)
	nFeatures := len(X[0])

	var newX [][]float64
	var newHeaders []string

	for _, row := range X {
		var newRow []float64
		for i := 0; i < nFeatures; i++ {
			newRow = append(newRow, row[i])
		}
		newX = append(newX, newRow)
	}
	newHeaders = append(newHeaders, headers...)

	// Only quadratic expansion for simplicity (degree 2)
	if degree == 2 {
		for i := 0; i < nFeatures; i++ {
			for j := i; j < nFeatures; j++ {
				for k := 0; k < nSamples; k++ {
					newX[k] = append(newX[k], X[k][i]*X[k][j])
				}
				if i == j {
					newHeaders = append(newHeaders, headers[i]+"^2")
				} else {
					newHeaders = append(newHeaders, headers[i]+"*"+headers[j])
				}
			}
		}
	}

	return newX, newHeaders
}
