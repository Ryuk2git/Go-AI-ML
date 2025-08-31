package stats

import "math"

type StandardScaler struct {
	Mean []float64
	Std  []float64
	fit  bool
}

// Standardizes each column  ti zero mean and unit variance
func StandardizeData(X [][]float64) [][]float64 {
	rows, cols := len(X), len(X[0])
	out := make([][]float64, rows)
	means := make([]float64, cols)
	stds := make([]float64, cols)
	for j := range cols {
		col := make([]float64, rows)
		for i := range rows {
			col[i] = X[i][j]
		}
		means[j] = Mean(col)
		stds[j] = Std(col)
	}

	for i := range rows {
		out[i] = make([]float64, cols)
		for j := range cols {
			if stds[j] != 0 {
				out[i][j] = (X[i][j] - means[j]) / stds[j]
			} else {
				out[i][j] = 0
			}
		}
	}

	return out

}

// MinMaxScale scales each column to [0, 1].
func MinMaxScale(X [][]float64) [][]float64 {
	rows, cols := len(X), len(X[0])
	out := make([][]float64, rows)
	mins := make([]float64, cols)
	maxs := make([]float64, cols)
	for j := range cols {
		col := make([]float64, rows)
		for i := range rows {
			col[i] = X[i][j]
		}
		mins[j], maxs[j] = MinMax(col)
	}
	for i := range rows {
		out[i] = make([]float64, cols)
		for j := range cols {
			if maxs[j] != mins[j] {
				out[i][j] = (X[i][j] - mins[j]) / (maxs[j] - mins[j])
			} else {
				out[i][j] = 0
			}
		}
	}
	return out
}

// RobustScale scales each column using median and IQR.
func RobustScale(X [][]float64) [][]float64 {
	rows, cols := len(X), len(X[0])
	out := make([][]float64, rows)
	medians := make([]float64, cols)
	iqrs := make([]float64, cols)
	for j := range cols {
		col := make([]float64, rows)
		for i := range rows {
			col[i] = X[i][j]
		}
		medians[j] = Median(col)
		iqrs[j] = Percentile(col, 75) - Percentile(col, 25)
	}
	for i := range rows {
		out[i] = make([]float64, cols)
		for j := range cols {
			if iqrs[j] != 0 {
				out[i][j] = (X[i][j] - medians[j]) / iqrs[j]
			} else {
				out[i][j] = 0
			}
		}
	}
	return out
}

func NewStandardScaler() *StandardScaler { return &StandardScaler{} }

func (s *StandardScaler) Fit(X [][]float64) error {
	if len(X) == 0 {
		return nil
	}
	r, c := len(X), len(X[0])
	s.Mean = make([]float64, c)
	s.Std = make([]float64, c)
	for j := 0; j < c; j++ {
		for i := 0; i < r; i++ {
			s.Mean[j] += X[i][j]
		}
		s.Mean[j] /= float64(r)
		v := 0.0
		for i := 0; i < r; i++ {
			d := X[i][j] - s.Mean[j]
			v += d * d
		}
		v /= float64(r)
		s.Std[j] = math.Sqrt(v)
		if s.Std[j] == 0 {
			s.Std[j] = 1
		}
	}
	s.fit = true
	return nil
}

func (s *StandardScaler) Transform(X [][]float64) [][]float64 {
	if !s.fit {
		return X
	}
	r, c := len(X), len(X[0])
	Y := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := make([]float64, c)
		for j := 0; j < c; j++ {
			row[j] = (X[i][j] - s.Mean[j]) / s.Std[j]
		}
		Y[i] = row
	}
	return Y
}

func (s *StandardScaler) FitTransform(X [][]float64) [][]float64 { _ = s.Fit(X); return s.Transform(X) }
