package stats

// ClipOutliers clips values in each column to the given lower and upper percentiles.
func ClipOutliers(X [][]float64, lower, upper float64) [][]float64 {
	rows, cols := len(X), len(X[0])
	out := make([][]float64, rows)
	lows := make([]float64, cols)
	highs := make([]float64, cols)
	for j := range cols {
		col := make([]float64, rows)
		for i := range rows {
			col[i] = X[i][j]
		}
		lows[j] = Percentile(col, lower)
		highs[j] = Percentile(col, upper)
	}
	for i := range rows {
		out[i] = make([]float64, cols)
		for j := range cols {
			v := X[i][j]
			if v < lows[j] {
				out[i][j] = lows[j]
			} else if v > highs[j] {
				out[i][j] = highs[j]
			} else {
				out[i][j] = v
			}
		}
	}
	return out
}
