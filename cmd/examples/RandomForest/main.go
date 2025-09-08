package main

import (
	"fmt"
	"math/rand"
	"time"

	"aiml/pkg/model"
)

// generateBinaryData creates a simple binary classification dataset.
// Rule: if x1 * x2 > 0 → class 1, else class 0.
func generateBinaryData(n int) (X [][]float64, y []int) {
	X = make([][]float64, n)
	y = make([]int, n)
	for i := 0; i < n; i++ {
		x1 := rand.Float64()*2 - 1 // [-1,1]
		x2 := rand.Float64()*2 - 1
		X[i] = []float64{x1, x2}
		if x1*x2 > 0 {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
	return
}

// trainTestSplit splits dataset into train/test sets with given ratio.
func trainTestSplit(X [][]float64, y []int, testRatio float64) (XTrain [][]float64, yTrain []int, XTest [][]float64, yTest []int) {
	n := len(X)
	indices := rand.Perm(n)
	testSize := int(float64(n) * testRatio)

	testIdx := indices[:testSize]
	trainIdx := indices[testSize:]

	for _, i := range trainIdx {
		XTrain = append(XTrain, X[i])
		yTrain = append(yTrain, y[i])
	}
	for _, i := range testIdx {
		XTest = append(XTest, X[i])
		yTest = append(yTest, y[i])
	}
	return
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== Random Forest Demo with Train/Test Split ===")

	// Step 1. Generate dataset
	X, y := generateBinaryData(1000)
	fmt.Printf("Generated %d samples with 2 features each.\n", len(X))
	fmt.Println("First 5 samples:")
	for i := 0; i < 5; i++ {
		fmt.Printf("  X=%v, y=%d\n", X[i], y[i])
	}

	// Step 2. Split into train/test sets
	XTrain, yTrain, XTest, yTest := trainTestSplit(X, y, 0.3)
	fmt.Printf("\nTrain size: %d, Test size: %d\n", len(XTrain), len(XTest))

	// Step 3. Initialize Random Forest
	rf := model.NewRandomForest(
		model.WithNEstimators(50), // more trees → better performance
		model.WithBootstrap(true),
	)
	fmt.Println("\nInitialized Random Forest with 50 trees, bootstrap enabled.")

	// Step 4. Train on training data
	fmt.Println("Training Random Forest...")
	if err := rf.Fit(XTrain, yTrain); err != nil {
		panic(fmt.Sprintf("training failed: %v", err))
	}
	fmt.Println("Training complete.")

	// Step 5. Predict on test data
	fmt.Println("\nMaking predictions on test data...")
	testPreds := rf.Predict(XTest)

	// Step 6. Show some example predictions
	fmt.Println("First 10 test predictions (X → Pred vs True):")
	for i := 0; i < 10 && i < len(XTest); i++ {
		fmt.Printf("  X=%v → Pred=%d, True=%d\n", XTest[i], testPreds[i], yTest[i])
	}

	// Step 7. Compute accuracy
	correct := 0
	for i := range yTest {
		if yTest[i] == testPreds[i] {
			correct++
		}
	}
	acc := float64(correct) / float64(len(yTest))
	fmt.Printf("\nFinal Accuracy on test data: %.2f%% (%d/%d)\n", acc*100, correct, len(yTest))
}
