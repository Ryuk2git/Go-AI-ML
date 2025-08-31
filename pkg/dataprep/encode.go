package dataprep

import "strconv"

// EncodeCategorical one-hot encodes a slice of string categories.
func EncodeCategorical(data []string) ([][]float64, map[string]int) {
	unique := map[string]int{}
	for _, v := range data {
		if _, ok := unique[v]; !ok {
			unique[v] = len(unique)
		}
	}
	out := make([][]float64, len(data))
	for i, v := range data {
		vec := make([]float64, len(unique))
		vec[unique[v]] = 1
		out[i] = vec
	}
	return out, unique
}

// LabelEncode encodes categories as integers.
func LabelEncode(data []string) ([]int, map[string]int) {
	unique := map[string]int{}
	out := make([]int, len(data))
	for i, v := range data {
		if _, ok := unique[v]; !ok {
			unique[v] = len(unique)
		}
		out[i] = unique[v]
	}
	return out, unique
}

// FrequencyEncode encodes categories by their frequency.
func FrequencyEncode(data []string) ([]float64, map[string]float64) {
	counts := map[string]float64{}
	for _, v := range data {
		counts[v]++
	}
	out := make([]float64, len(data))
	for i, v := range data {
		out[i] = counts[v] / float64(len(data))
	}
	return out, counts
}

// EncodeCategoricalAll encodes all categorical columns in a dataset.
// method: "label", "onehot", or "freq"
func EncodeCategoricalAll(data [][]string, headers []string, method string) ([][]float64, map[string]interface{}) {
	nRows := len(data)
	encoded := [][]float64{}
	encoders := make(map[string]interface{})

	for j, header := range headers {
		col := make([]string, nRows)
		for i := 0; i < nRows; i++ {
			col[i] = data[i][j]
		}

		// Try numeric check
		isNumeric := true
		for _, v := range col {
			if _, err := strconv.ParseFloat(v, 64); err != nil {
				isNumeric = false
				break
			}
		}

		if isNumeric {
			// Keep numeric column
			nums := make([]float64, nRows)
			for i, v := range col {
				f, _ := strconv.ParseFloat(v, 64)
				nums[i] = f
			}
			// append column as single feature
			if len(encoded) == 0 {
				for i := 0; i < nRows; i++ {
					encoded = append(encoded, []float64{})
				}
			}
			for i := 0; i < nRows; i++ {
				encoded[i] = append(encoded[i], nums[i])
			}
		} else {
			// Apply chosen encoding
			switch method {
			case "onehot":
				oh, mapping := EncodeCategorical(col)
				encoders[header] = mapping
				if len(encoded) == 0 {
					encoded = oh
				} else {
					for i := 0; i < nRows; i++ {
						encoded[i] = append(encoded[i], oh[i]...)
					}
				}
			case "freq":
				freq, mapping := FrequencyEncode(col)
				encoders[header] = mapping
				if len(encoded) == 0 {
					for i := 0; i < nRows; i++ {
						encoded = append(encoded, []float64{freq[i]})
					}
				} else {
					for i := 0; i < nRows; i++ {
						encoded[i] = append(encoded[i], freq[i])
					}
				}
			case "label":
				fallthrough
			default:
				labels, mapping := LabelEncode(col)
				encoders[header] = mapping
				if len(encoded) == 0 {
					for i := 0; i < nRows; i++ {
						encoded = append(encoded, []float64{float64(labels[i])})
					}
				} else {
					for i := 0; i < nRows; i++ {
						encoded[i] = append(encoded[i], float64(labels[i]))
					}
				}
			}
		}
	}

	return encoded, encoders
}
