package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	"aiml/pkg/data"
	"aiml/pkg/model"
	"aiml/pkg/optim"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// --- Supervised: Linear Regression Data Generation ---

// generateLinearData creates n samples with d features and a linear relationship plus noise.
func generateLinearData(n, d int) (X [][]float64, y []float64, trueW []float64, trueB float64) {
	trueW = make([]float64, d)
	for i := range trueW {
		trueW[i] = rand.Float64()*4 - 2 // weights in [-2,2]
	}
	trueB = rand.Float64()*2 - 1 // bias in [-1,1]
	X = make([][]float64, n)
	y = make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			X[i][j] = rand.Float64()*10 - 5 // features in [-5,5]
		}
		// y = w^T x + b + noise
		y[i] = trueB
		for j := 0; j < d; j++ {
			y[i] += trueW[j] * X[i][j]
		}
		y[i] += rand.NormFloat64() * 0.5 // add noise
	}
	return
}

// generateComplexRegressionData creates a high-dimensional, complex, and noisy dataset.
func generateComplexRegressionData(n, d int) (X [][]float64, y []float64, trueW []float64, trueB float64) {
	trueW = make([]float64, d)
	for i := range trueW {
		trueW[i] = rand.Float64()*4 - 2 // weights in [-2,2]
	}
	trueB = rand.Float64()*2 - 1 // bias in [-1,1]
	X = make([][]float64, n)
	y = make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			X[i][j] = rand.NormFloat64() * 5 // features: normal distribution
		}
		// Linear + nonlinear terms
		y[i] = trueB
		for j := 0; j < d; j++ {
			y[i] += trueW[j] * X[i][j]
			if j%2 == 0 {
				y[i] += 0.5 * math.Sin(X[i][j]) // add nonlinearity
			}
			if j%3 == 0 {
				y[i] += 0.2 * X[i][j] * X[i][j] // quadratic term
			}
		}
		y[i] += rand.NormFloat64() * 2 // more noise
	}
	return
}

// plotLinearRegression visualizes the data and the learned regression line by plotting the first feature.
func plotLinearRegression(X [][]float64, y []float64, learnedW float64, learnedB float64, filename string) {
	p := plot.New()
	p.Title.Text = "Linear Regression on First Feature"
	p.X.Label.Text = "Feature 1"
	p.Y.Label.Text = "Target Y"

	// Plot the data points.
	pts := make(plotter.XYs, len(X))
	for i := range X {
		pts[i].X = X[i][0]
		pts[i].Y = y[i]
	}
	s, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	s.Color = color.RGBA{B: 255, A: 255, R: 50, G: 50}
	p.Add(s)

	// Plot the learned regression line based on the first feature.
	linePts := plotter.XYs{
		{X: -5, Y: learnedW*(-5) + learnedB},
		{X: 5, Y: learnedW*5 + learnedB},
	}
	l, err := plotter.NewLine(linePts)
	if err != nil {
		log.Fatal(err)
	}
	l.Color = color.RGBA{R: 255, A: 255}
	l.LineStyle.Width = vg.Points(3)
	p.Add(l)

	// Save the plot to a file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, filename); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Saved linear regression plot to %s\n", filename)
}

// --- Unsupervised: KMeans Data Generation and Model ---

// generateClusterData creates n samples in k clusters, each with d features.
func generateClusterData(n, k, d int) (X [][]float64, trueCenters [][]float64) {
	X = make([][]float64, n)
	trueCenters = make([][]float64, k)
	for i := 0; i < k; i++ {
		center := make([]float64, d)
		for j := 0; j < d; j++ {
			center[j] = rand.Float64()*10 - 5
		}
		trueCenters[i] = center
	}
	for i := 0; i < n; i++ {
		c := rand.Intn(k)
		x := make([]float64, d)
		for j := 0; j < d; j++ {
			x[j] = trueCenters[c][j] + rand.NormFloat64()
		}
		X[i] = x
	}
	return
}

// generateComplexClusterData creates a high-dimensional, complex, and noisy dataset.
func generateComplexClusterData(n, k, d int) (X [][]float64, trueCenters [][]float64) {
	X = make([][]float64, n)
	trueCenters = make([][]float64, k)
	for i := 0; i < k; i++ {
		center := make([]float64, d)
		for j := 0; j < d; j++ {
			center[j] = rand.Float64()*30 - 15 // spread out centers
		}
		trueCenters[i] = center
	}
	for i := 0; i < n; i++ {
		c := rand.Intn(k)
		x := make([]float64, d)
		for j := 0; j < d; j++ {
			x[j] = trueCenters[c][j] + rand.NormFloat64()*5 // more overlap/noise
			if rand.Float64() < 0.1 {
				x[j] += rand.NormFloat64() * 20 // outlier
			}
		}
		X[i] = x
	}
	return
}

// --- Minimal KMeans Implementation for Demo ---
type KMeans struct {
	K        int
	Centers  [][]float64
	MaxIters int
}

func NewKMeans(k, maxIters int) *KMeans {
	return &KMeans{K: k, MaxIters: maxIters}
}

func (km *KMeans) Fit(X [][]float64) {
	n, d := len(X), len(X[0])
	km.Centers = make([][]float64, km.K)
	perm := rand.Perm(n)
	for i := 0; i < km.K; i++ {
		km.Centers[i] = make([]float64, d)
		copy(km.Centers[i], X[perm[i]])
	}
	assign := make([]int, n)
	for iter := 0; iter < km.MaxIters; iter++ {
		changed := false
		// Assign step
		for i := 0; i < n; i++ {
			minDist := math.MaxFloat64
			for k := 0; k < km.K; k++ {
				dist := 0.0
				for j := 0; j < d; j++ {
					dx := X[i][j] - km.Centers[k][j]
					dist += dx * dx
				}
				if dist < minDist {
					minDist = dist
					assign[i] = k
				}
			}
		}
		// Update step
		counts := make([]int, km.K)
		newCenters := make([][]float64, km.K)
		for k := 0; k < km.K; k++ {
			newCenters[k] = make([]float64, d)
		}
		for i := 0; i < n; i++ {
			k := assign[i]
			counts[k]++
			for j := 0; j < d; j++ {
				newCenters[k][j] += X[i][j]
			}
		}
		for k := 0; k < km.K; k++ {
			if counts[k] > 0 {
				for j := 0; j < d; j++ {
					newCenters[k][j] /= float64(counts[k])
				}
			}
			if !equalSlice(newCenters[k], km.Centers[k]) {
				changed = true
			}
		}
		km.Centers = newCenters
		if !changed {
			break
		}
	}
}

func (km *KMeans) Predict(x []float64) int {
	minDist := math.MaxFloat64
	best := 0
	for k, c := range km.Centers {
		dist := 0.0
		for j := range x {
			dx := x[j] - c[j]
			dist += dx * dx
		}
		if dist < minDist {
			minDist = dist
			best = k
		}
	}
	return best
}

func equalSlice(a, b []float64) bool {
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-8 {
			return false
		}
	}
	return true
}

// plotKMeans visualizes the clustered data and the final centroids by plotting the first two features.
func plotKMeans(X [][]float64, assignments []int, centers [][]float64, filename string) {
	p := plot.New()
	p.Title.Text = "K-Means Clustering on First Two Features"
	p.X.Label.Text = "Feature 1"
	p.Y.Label.Text = "Feature 2"

	// Define colors for the clusters
	colors := []color.RGBA{
		{R: 255, A: 255},
		{G: 255, A: 255},
		{B: 255, A: 255},
	}
	if len(assignments) > 0 {
		numClusters := 0
		for _, a := range assignments {
			if a >= numClusters {
				numClusters = a + 1
			}
		}
		// Plot data points colored by their cluster assignment
		for k := 0; k < numClusters; k++ {
			pts := make(plotter.XYs, 0)
			for i, assignment := range assignments {
				if assignment == k {
					// Check if there are at least two features to plot
					if len(X[i]) >= 2 {
						pts = append(pts, plotter.XY{X: X[i][0], Y: X[i][1]})
					}
				}
			}
			s, err := plotter.NewScatter(pts)
			if err != nil {
				log.Fatal(err)
			}
			s.Color = colors[k%len(colors)]
			p.Add(s)
		}
	}

	// Plot the centroids
	centroidPts := make(plotter.XYs, len(centers))
	for i, c := range centers {
		if len(c) >= 2 {
			centroidPts[i] = plotter.XY{X: c[0], Y: c[1]}
		}
	}
	c, err := plotter.NewScatter(centroidPts)
	if err != nil {
		log.Fatal(err)
	}
	c.Color = color.RGBA{A: 255}
	c.Shape = draw.CrossGlyph{}
	c.Radius = vg.Points(5)
	p.Add(c)

	if err := p.Save(4*vg.Inch, 4*vg.Inch, filename); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Saved K-Means plot to %s\n", filename)
}

// --- Main Demo ---

func main() {
	rand.Seed(time.Now().UnixNano())

	// --- Supervised Learning: Simple Demo with Plotting ---
	fmt.Println("=== Supervised Learning: Linear Regression Simple Demo ===")
	start := time.Now()
	n, d := 500, 5
	X, y, trueW, trueB := generateLinearData(n, d)
	fmt.Printf("Generated %d samples with %d features.\n", n, d)

	modelLR := model.NewLinearRegression(d, 0.01, 10, 50)
	optimizer := optim.NewSGD(0.01)

	batchSize := 50
	batchChan := make(chan data.Batch)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer close(batchChan)
		for i := 0; i < n; i += batchSize {
			end := i + batchSize
			if end > n {
				end = n
			}
			batch := data.Batch{
				X: X[i:end],
				Y: y[i:end],
			}
			batchChan <- batch
		}
		wg.Done()
	}()

	modelLR.Fit(batchChan, 10, optimizer)
	wg.Wait()
	fmt.Printf("Training completed in %v\n", time.Since(start))
	fmt.Printf("True weights: %v, bias: %.3f\n", trueW, trueB)
	fmt.Printf("Learned weights: %v, bias: %.3f\n", modelLR.W, modelLR.Bias())
	plotLinearRegression(X, y, modelLR.W[0], modelLR.Bias(), "linear_regression_simple.png")
	fmt.Println()

	// --- Unsupervised Learning: Simple Demo with Plotting ---
	fmt.Println("=== Unsupervised Learning: KMeans Clustering Simple Demo ===")
	start = time.Now()
	n, k, d := 300, 3, 2
	Xc, trueCenters := generateClusterData(n, k, d)
	fmt.Printf("Generated %d samples in %d clusters with %d features.\n", n, k, d)

	kmeans := NewKMeans(k, 100)
	kmeans.Fit(Xc)
	fmt.Printf("KMeans training completed in %v\n", time.Since(start))
	fmt.Println("True centers (approx):")
	for _, c := range trueCenters {
		fmt.Printf("  %v\n", c)
	}
	fmt.Println("Learned centers:")
	for _, c := range kmeans.Centers {
		fmt.Printf("  %v\n", c)
	}
	assignments := make([]int, n)
	for i, x := range Xc {
		assignments[i] = kmeans.Predict(x)
	}
	plotKMeans(Xc, assignments, kmeans.Centers, "kmeans_clusters_simple.png")
	fmt.Println()

	// --- Supervised Learning: Stress Test with Plotting ---
	fmt.Println("=== Supervised Learning: Linear Regression Stress Test ===")
	start = time.Now()
	n, d = 20000, 20 // Large n, high d
	Xcomplex, ycomplex, _, _ := generateComplexRegressionData(n, d)
	fmt.Printf("Generated %d samples with %d features (complex, noisy).\n", n, d)

	modelLRstress := model.NewLinearRegression(d, 0.001, 5, 256)
	optimizerstress := optim.NewSGD(0.001)

	batchSizeStress := 256
	batchChanStress := make(chan data.Batch)
	var wgStress sync.WaitGroup
	wgStress.Add(1)
	go func() {
		defer close(batchChanStress)
		for i := 0; i < n; i += batchSizeStress {
			end := i + batchSizeStress
			if end > n {
				end = n
			}
			batch := data.Batch{
				X: Xcomplex[i:end],
				Y: ycomplex[i:end],
			}
			batchChanStress <- batch
		}
		wgStress.Done()
	}()

	modelLRstress.Fit(batchChanStress, 5, optimizerstress)
	wgStress.Wait()
	fmt.Printf("Training completed in %v\n", time.Since(start))
	fmt.Printf("Learned weights (first 5): %v, bias: %.3f\n", modelLRstress.W[:5], modelLRstress.Bias())

	plotLinearRegression(Xcomplex, ycomplex, modelLRstress.W[0], modelLRstress.Bias(), "linear_regression_complex.png")
	fmt.Println()

	// --- Unsupervised Learning: Stress Test with Plotting ---
	fmt.Println("=== Unsupervised Learning: KMeans Stress Test ===")
	start = time.Now()
	n, k, d = 10000, 3, 10 // Large n, more clusters, higher d
	Xcstress, _ := generateComplexClusterData(n, k, d)
	fmt.Printf("Generated %d samples in %d clusters with %d features (complex, noisy).\n", n, k, d)

	kmeansStress := NewKMeans(k, 100)
	kmeansStress.Fit(Xcstress)
	fmt.Printf("KMeans training completed in %v\n", time.Since(start))
	fmt.Println("Learned centers (first 2):")
	for _, c := range kmeansStress.Centers[:2] {
		fmt.Printf("  %v\n", c[:2])
	}
	assignmentsStress := make([]int, n)
	for i, x := range Xcstress {
		assignmentsStress[i] = kmeansStress.Predict(x)
	}
	plotKMeans(Xcstress, assignmentsStress, kmeansStress.Centers, "kmeans_clusters_complex.png")
	fmt.Println()
}
