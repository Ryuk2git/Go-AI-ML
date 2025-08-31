package model

// Model is a generic supervised learning interface.
type Model interface {
	Fit(X [][]float64, y []float64) error
	Predict(X [][]float64) []float64
}

// Classifier optionally exposes probabilities.
type Classifier interface {
	Model
	PredictProba(X [][]float64) []float64 // returns p(y=1) for binary classifiers
}

// Clusterer is for unsupervised clustering.
type Clusterer interface {
	Fit(X [][]float64) error
	Predict(X [][]float64) []int // cluster assignments
}

// Transformer is for preprocessing steps (fit on train, transform both).
type Transformer interface {
	Fit(X [][]float64) error
	Transform(X [][]float64) [][]float64
	FitTransform(X [][]float64) [][]float64
}
