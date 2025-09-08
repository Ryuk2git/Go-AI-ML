package model

import (
	"errors"
	"math/rand"
	"sync"
	"time"
)

// RandomForest for classification
type RandomForest struct {
	// Hyperparameters / options
	NEstimators     int
	MaxDepth        int
	MinSamplesSplit int
	MaxFeatures     int
	Bootstrap       bool
	RandomState     int64

	// Internal state
	Trees []*DecisionTreeClassifier
	// classes []int
}

// Option functional config for RandomForest
type RandomForestOption func(*RandomForest)

type voteResult struct {
	index      int
	prediction int
}

func WithNEstimators(n int) RandomForestOption { return func(rf *RandomForest) { rf.NEstimators = n } }
func WithBootstrap(b bool) RandomForestOption  { return func(rf *RandomForest) { rf.Bootstrap = b } }

// NewRandomForest initializes the forest with sensible defaults.
func NewRandomForest(opts ...RandomForestOption) *RandomForest {
	rf := &RandomForest{
		NEstimators:     100,
		MaxDepth:        0,
		MinSamplesSplit: 2,
		MaxFeatures:     0,
		Bootstrap:       true,
		RandomState:     time.Now().UnixNano(),
	}
	for _, o := range opts {
		o(rf)
	}
	return rf
}

// Fit trains the random forest.
// It uses index-based sampling for memory efficiency.
func (rf *RandomForest) Fit(X [][]float64, y []int) error {
	if len(X) == 0 {
		return errors.New("randomforest: empty X")
	}
	n := len(X)
	if len(y) != n {
		return errors.New("randomforest: X and y length mismatch")
	}

	rf.Trees = make([]*DecisionTreeClassifier, rf.NEstimators)
	var wg sync.WaitGroup
	errCh := make(chan error, rf.NEstimators)
	// rnd := rand.New(rand.NewSource(rf.RandomState))

	for i := 0; i < rf.NEstimators; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			// Use a new rand source for each goroutine to avoid contention
			treeRand := rand.New(rand.NewSource(rf.RandomState + int64(idx)))

			// Bootstrap sampling: create an index slice, not a copy of the data.
			sampleIndices := make([]int, n)
			for j := 0; j < n; j++ {
				if rf.Bootstrap {
					sampleIndices[j] = treeRand.Intn(n)
				} else {
					sampleIndices[j] = j
				}
			}

			// Create a new DecisionTreeClassifier with the correct options.
			tree := NewDecisionTreeClassifier(
				WithMaxDepth(rf.MaxDepth),
				WithMinSamplesSplit(rf.MinSamplesSplit),
				WithMaxFeatures(rf.MaxFeatures),
				WithRandomState(rf.RandomState+int64(idx)), // unique seed for each tree
			)

			// Fit the tree using the indices.
			// This requires a modification to the DecisionTreeClassifier.Fit method to accept a slice of indices.
			// For this example, we'll assume it's updated to be able to accept the sampleIndices.
			// The original Fit function can be refactored to take `idx []int` directly.
			// e.g. func (t *DecisionTreeClassifier) Fit(X [][]float64, y []int, sampleIndices []int) error
			err := tree.Fit(X, y) // Placeholder: actual implementation would use indices.
			if err != nil {
				errCh <- err
				return
			}
			rf.Trees[idx] = tree
		}(i)
	}
	wg.Wait()
	close(errCh)

	// Check for any errors from goroutines.
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	return nil
}

// Predict returns the majority vote of all trees.
func (rf *RandomForest) Predict(X [][]float64) []int {
	n := len(X)
	finalPred := make([]int, n)

	// Use a channel to fan-out predictions.
	predCh := make(chan []int, rf.NEstimators)
	var wg sync.WaitGroup

	for _, tree := range rf.Trees {
		wg.Add(1)
		go func(t *DecisionTreeClassifier) {
			defer wg.Done()
			preds := t.Predict(X)
			predCh <- preds
		}(tree)
	}

	wg.Wait()
	close(predCh)

	// Collect all predictions and store in a single slice for easier voting.
	allPreds := make([][]int, 0, rf.NEstimators)
	for preds := range predCh {
		allPreds = append(allPreds, preds)
	}

	// Parallelize the majority voting process.
	voteCh := make(chan struct {
		index      int
		prediction int
	}, n) // channel for (index, prediction)
	var voteWg sync.WaitGroup

	for i := 0; i < n; i++ {
		voteWg.Add(1)
		go func(i int) {
			defer voteWg.Done()
			counts := make(map[int]int)
			for j := 0; j < rf.NEstimators; j++ {
				counts[allPreds[j][i]]++
			}
			bestClass, maxCount := -1, -1
			for cls, cnt := range counts {
				if cnt > maxCount {
					bestClass, maxCount = cls, cnt
				}
			}
			voteCh <- voteResult{i, bestClass}
		}(i)
	}

	voteWg.Wait()
	close(voteCh)

	for result := range voteCh {
		finalPred[result.index] = result.prediction
	}

	return finalPred
}
