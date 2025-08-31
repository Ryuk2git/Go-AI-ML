package loader

import "math/rand"

// TrainTestSplit splits X, Y into train and test sets by ratio.
func TrainTestSplit(X [][]float64, Y []float64, testRatio float64) (XTrain, XTest [][]float64, YTrain, YTest []float64) {
	n := len(X)
	indices := rand.Perm(n)
	nTest := int(float64(n) * testRatio)
	for i := range n {
		if i < nTest {
			XTest = append(XTest, X[indices[i]])
			YTest = append(YTest, Y[indices[i]])
		} else {
			XTrain = append(XTrain, X[indices[i]])
			YTrain = append(YTrain, Y[indices[i]])
		}
	}
	return
}

// ShuffleData shuffles X and Y in unison.
func ShuffleData(X [][]float64, Y []float64) ([][]float64, []float64) {
	n := len(X)
	indices := rand.Perm(n)
	XShuf := make([][]float64, n)
	YShuf := make([]float64, n)
	for i, idx := range indices {
		XShuf[i] = X[idx]
		YShuf[i] = Y[idx]
	}
	return XShuf, YShuf
}

// KFoldSplit yields k folds of train/test indices.
func KFoldSplit(n, k int) [][]int {
	indices := rand.Perm(n)
	folds := make([][]int, k)
	for i := range n {
		folds[i%k] = append(folds[i%k], indices[i])
	}
	return folds
}
