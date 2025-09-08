package main

import (
	"fmt"
	"math/rand"
	"time"

	"aiml/pkg/model"
)

// generateClassificationData creates a synthetic dataset with Gaussian blobs
func generateClassificationData(nSamples, nFeatures, nClasses int) ([][]float64, []int) {
	X := make([][]float64, nSamples)
	y := make([]int, nSamples)

	// Random centers for each class
	centers := make([][]float64, nClasses)
	for i := 0; i < nClasses; i++ {
		centers[i] = make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			centers[i][j] = rand.Float64()*10 - 5 // random center between -5 and 5
		}
	}

	// Assign points to clusters with Gaussian noise
	for i := 0; i < nSamples; i++ {
		class := rand.Intn(nClasses)
		X[i] = make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			X[i][j] = centers[class][j] + rand.NormFloat64() // noise around center
		}
		y[i] = class
	}

	return X, y
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Generate dataset
	X, y := generateClassificationData(5000, 4, 4)

	// Train decision tree
	tree := model.NewDecisionTreeClassifier(model.WithMaxDepth(5), model.WithMinSamplesSplit(2)) // maxDepth=5, minSamplesSplit=2
	tree.Fit(X, y)

	// Test predictions on first 10 samples
	fmt.Println("Testing Decision Tree on Synthetic Dataset:")
	for i := 0; i < 10; i++ {
		pred := tree.Predict([][]float64{X[i]})[0]
		fmt.Printf("Sample: %v, True Label: %d, Predicted: %d\n",
			X[i], y[i], pred)
	}
}
