package model

import (
	"bytes"
	"encoding/gob"
	"errors"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// ---------------------------
// Types & options
// ---------------------------

// DecisionTreeClassifier is a CART-style classifier.
type DecisionTreeClassifier struct {
	// Hyperparameters / options
	MaxDepth            int     // maximum depth (root depth = 0). 0 => no limit
	MinSamplesSplit     int     // minimum samples to attempt a split
	MinSamplesLeaf      int     // minimum samples required in each leaf
	Criterion           string  // "gini" (default) or "entropy"
	MaxFeatures         int     // 0 => use all features, >0 => number of features to sample when looking for split
	MinImpurityDecrease float64 // minimal impurity decrease to accept a split
	RandomState         int64   // seed for randomness (feature subsampling)

	// internals
	root    *dtNode
	classes []int // unique class labels (order used by probas)
}

// dtNode holds a node in the tree.
type dtNode struct {
	// internal node fields
	isLeaf    bool
	feature   int
	threshold float64 // numeric threshold: x <= threshold => left
	isCat     bool    // true if this split is a categorical equality split (x == threshold)
	left      *dtNode
	right     *dtNode

	// leaf data
	n         int
	probas    []float64 // probability distribution across classes (aligned with tree.classes)
	predIndex int       // index into classes for predicted class (majority)
}

// Option functional config
type Option func(*DecisionTreeClassifier)

func WithMaxDepth(d int) Option { return func(t *DecisionTreeClassifier) { t.MaxDepth = d } }
func WithMinSamplesSplit(n int) Option {
	return func(t *DecisionTreeClassifier) { t.MinSamplesSplit = n }
}
func WithMinSamplesLeaf(n int) Option {
	return func(t *DecisionTreeClassifier) { t.MinSamplesLeaf = n }
}
func WithCriterion(c string) Option { return func(t *DecisionTreeClassifier) { t.Criterion = c } }
func WithMaxFeatures(k int) Option  { return func(t *DecisionTreeClassifier) { t.MaxFeatures = k } }
func WithMinImpurityDecrease(v float64) Option {
	return func(t *DecisionTreeClassifier) { t.MinImpurityDecrease = v }
}
func WithRandomState(seed int64) Option {
	return func(t *DecisionTreeClassifier) { t.RandomState = seed }
}

// NewDecisionTreeClassifier returns a classifier with sensible defaults.
func NewDecisionTreeClassifier(opts ...Option) *DecisionTreeClassifier {
	d := &DecisionTreeClassifier{
		MaxDepth:            0, // 0 => no explicit max (stopping by other criteria)
		MinSamplesSplit:     2,
		MinSamplesLeaf:      1,
		Criterion:           "gini",
		MaxFeatures:         0,
		MinImpurityDecrease: 0.0,
		RandomState:         time.Now().UnixNano(),
	}
	for _, o := range opts {
		o(d)
	}
	return d
}

// ---------------------------
// Public API: Fit / Predict / PredictProba / Prune / Save/Load
// ---------------------------

// Fit trains the decision tree on X (n x p) and y (n labels as ints).
// X must be length n; each row should have same number of columns.
// Missing values must be math.NaN(). Categorical features:
// encode categories as integers (0,1,2...) in the corresponding float64 entry.
func (t *DecisionTreeClassifier) Fit(X [][]float64, y []int) error {
	if len(X) == 0 {
		return errors.New("dtree: empty X")
	}
	n := len(X)
	if len(y) != n {
		return errors.New("dtree: X and y length mismatch")
	}
	p := len(X[0])
	for i := range X {
		if len(X[i]) != p {
			return errors.New("dtree: inconsistent number of features in X rows")
		}
	}

	// collect classes and build class list
	classMap := map[int]int{}
	t.classes = nil
	for _, lab := range y {
		if _, ok := classMap[lab]; !ok {
			classMap[lab] = len(t.classes)
			t.classes = append(t.classes, lab)
		}
	}
	if len(t.classes) == 0 {
		return errors.New("dtree: no classes in y")
	}

	// indices of samples currently considered
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		idx[i] = i
	}

	rnd := rand.New(rand.NewSource(t.RandomState))

	// impurity helper
	impurityFunc := func(counts []int) float64 {
		if t.Criterion == "entropy" {
			return entropyFromCounts(counts)
		}
		return giniFromCounts(counts)
	}

	t.root = t.buildNode(X, y, idx, 0, p, len(t.classes), impurityFunc, rnd)
	return nil
}

// Predict returns predicted class labels aligned with the labels the tree was trained on.
func (t *DecisionTreeClassifier) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i := range X {
		probs := t.predictProbaSingle(X[i])
		// index of max prob
		maxIdx := 0
		for j := 1; j < len(probs); j++ {
			if probs[j] > probs[maxIdx] {
				maxIdx = j
			}
		}
		out[i] = t.classes[maxIdx]
	}
	return out
}

// PredictProba returns the per-class probability vectors for rows in X.
func (t *DecisionTreeClassifier) PredictProba(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i := range X {
		out[i] = t.predictProbaSingle(X[i])
	}
	return out
}

// PruneReducedError performs reduced-error post-pruning using validation data (Xval,yval).
// It will attempt to prune internal nodes if pruning improves accuracy on validation set.
// Returns number of pruned nodes.
func (t *DecisionTreeClassifier) PruneReducedError(Xval [][]float64, yval []int) (int, error) {
	if t.root == nil {
		return 0, errors.New("dtree: tree not trained")
	}
	if len(Xval) == 0 || len(yval) != len(Xval) {
		return 0, errors.New("dtree: invalid validation set")
	}
	// Compute baseline accuracy
	baseline := accuracyInt(yval, t.Predict(Xval))
	pruned := t.pruneNodeReducedError(t.root, Xval, yval, baseline)
	return pruned, nil
}

// MarshalBinary implements encoding.BinaryMarshaler using gob.
func (t *DecisionTreeClassifier) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	// encode basic struct (excluding function pointers)
	if err := enc.Encode(t.MaxDepth); err != nil {
		return nil, err
	}
	if err := enc.Encode(t.MinSamplesSplit); err != nil {
		return nil, err
	}
	if err := enc.Encode(t.MinSamplesLeaf); err != nil {
		return nil, err
	}
	if err := enc.Encode(t.Criterion); err != nil {
		return nil, err
	}
	if err := enc.Encode(t.MaxFeatures); err != nil {
		return nil, err
	}
	if err := enc.Encode(t.MinImpurityDecrease); err != nil {
		return nil, err
	}
	if err := enc.Encode(t.RandomState); err != nil {
		return nil, err
	}
	if err := enc.Encode(t.classes); err != nil {
		return nil, err
	}
	// encode tree recursively
	if err := enc.Encode(t.root); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler using gob.
func (t *DecisionTreeClassifier) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&t.MaxDepth); err != nil {
		return err
	}
	if err := dec.Decode(&t.MinSamplesSplit); err != nil {
		return err
	}
	if err := dec.Decode(&t.MinSamplesLeaf); err != nil {
		return err
	}
	if err := dec.Decode(&t.Criterion); err != nil {
		return err
	}
	if err := dec.Decode(&t.MaxFeatures); err != nil {
		return err
	}
	if err := dec.Decode(&t.MinImpurityDecrease); err != nil {
		return err
	}
	if err := dec.Decode(&t.RandomState); err != nil {
		return err
	}
	if err := dec.Decode(&t.classes); err != nil {
		return err
	}
	if err := dec.Decode(&t.root); err != nil {
		return err
	}
	return nil
}

// ---------------------------
// Internal builders & helpers
// ---------------------------

// A struct to hold the results of a single feature's best split search.
type splitResult struct {
	gain      float64
	feature   int
	threshold float64
	isCat     bool
	leftIdx   []int
	rightIdx  []int
}

// pair is a named type for a value and its original index.
type pair struct {
	v float64
	i int
}

func (t *DecisionTreeClassifier) buildNode(X [][]float64, y []int, idx []int, depth, p, nClasses int, impurity func([]int) float64, rnd *rand.Rand) *dtNode {
	node := &dtNode{n: len(idx)}

	// compute class counts
	counts := make([]int, nClasses)
	for _, ii := range idx {
		ci := classIndex(y[ii], t.classes)
		counts[ci]++
	}
	// make leaf if pure or too few samples or depth reached
	if isPure(counts) || (t.MinSamplesSplit > 0 && len(idx) < t.MinSamplesSplit) {
		node.isLeaf = true
		node.probas = countsToProbas(counts)
		node.predIndex = argmax(counts)
		return node
	}
	if t.MaxDepth > 0 && depth >= t.MaxDepth {
		node.isLeaf = true
		node.probas = countsToProbas(counts)
		node.predIndex = argmax(counts)
		return node
	}

	// determine features to try
	featIndices := make([]int, p)
	for j := 0; j < p; j++ {
		featIndices[j] = j
	}
	if t.MaxFeatures > 0 && t.MaxFeatures < p {
		for i := 0; i < p; i++ {
			j := i + rnd.Intn(p-i)
			featIndices[i], featIndices[j] = featIndices[j], featIndices[i]
		}
		featIndices = featIndices[:t.MaxFeatures]
	}

	parentImpurity := impurity(counts)
	bestGain := 0.0
	var bestResult splitResult

	// Channel to receive results from goroutines.
	results := make(chan splitResult, len(featIndices))
	var wg sync.WaitGroup

	// Parallel search for the best split for each feature.
	for _, f := range featIndices {
		wg.Add(1)
		go func(f int) {
			defer wg.Done()
			result := t.findBestSplitForFeature(X, y, idx, f, nClasses, parentImpurity, impurity)
			results <- result
		}(f)
	}

	// Wait for all goroutines to finish.
	wg.Wait()
	close(results)

	// Collect results from the channel and find the best overall split.
	for result := range results {
		if result.gain > bestGain {
			bestGain = result.gain
			bestResult = result
		}
	}

	// Decide whether to split
	if bestResult.feature == -1 || bestGain <= t.MinImpurityDecrease {
		node.isLeaf = true
		node.probas = countsToProbas(counts)
		node.predIndex = argmax(counts)
		return node
	}

	// found a valid split; create internal node
	node.isLeaf = false
	node.feature = bestResult.feature
	node.threshold = bestResult.threshold
	node.isCat = bestResult.isCat
	// build children recursively
	node.left = t.buildNode(X, y, bestResult.leftIdx, depth+1, p, nClasses, impurity, rnd)
	node.right = t.buildNode(X, y, bestResult.rightIdx, depth+1, p, nClasses, impurity, rnd)
	return node
}

// findBestSplitForFeature is a goroutine-safe helper that finds the best split for a single feature.
func (t *DecisionTreeClassifier) findBestSplitForFeature(X [][]float64, y []int, idx []int, f, nClasses int, parentImpurity float64, impurity func([]int) float64) splitResult {
	result := splitResult{gain: 0.0, feature: -1}

	// build pairs of (value,index)
	tmp := make([]pair, 0, len(idx))
	for _, ii := range idx {
		tmp = append(tmp, pair{X[ii][f], ii})
	}

	// handle missing values: separate NaNs
	nans := make([]int, 0)
	valid := make([]pair, 0, len(tmp))
	for _, pval := range tmp {
		if math.IsNaN(pval.v) {
			nans = append(nans, pval.i)
		} else {
			valid = append(valid, pval)
		}
	}

	// if too few non-NaN samples, skip this feature
	if len(valid) == 0 {
		return result
	}

	// try categorical-equality splits if values are integer-like and small unique set
	uniqueVals := uniqueValuesFromPairs(valid)
	tryCat := false
	if len(uniqueVals) <= 30 {
		intLike := true
		for _, v := range uniqueVals {
			if !almostInt(v) {
				intLike = false
				break
			}
		}
		if intLike {
			tryCat = true
		}
	}

	if tryCat {
		for _, uv := range uniqueVals {
			leftIdx := make([]int, 0, len(idx))
			rightIdx := make([]int, 0, len(idx))
			for _, pval := range valid {
				if pval.v == uv {
					leftIdx = append(leftIdx, pval.i)
				} else {
					rightIdx = append(rightIdx, pval.i)
				}
			}
			// try NaNs on left
			leftWithNaN := append(append([]int(nil), leftIdx...), nans...)
			rightWithNaN := append([]int(nil), rightIdx...)
			if okSplit(leftWithNaN, rightWithNaN, t.MinSamplesLeaf) {
				leftCounts := countsFromIndices(y, leftWithNaN, nClasses, t.classes)
				rightCounts := countsFromIndices(y, rightWithNaN, nClasses, t.classes)
				impL := impurity(leftCounts)
				impR := impurity(rightCounts)
				weighted := (float64(len(leftWithNaN))/float64(len(idx)))*impL + (float64(len(rightWithNaN))/float64(len(idx)))*impR
				gain := parentImpurity - weighted
				if gain > result.gain {
					result = splitResult{gain: gain, feature: f, threshold: uv, isCat: true, leftIdx: leftWithNaN, rightIdx: rightWithNaN}
				}
			}
			// try NaNs on right
			leftNoNaN := append([]int(nil), leftIdx...)
			rightWithNaN2 := append(append([]int(nil), rightIdx...), nans...)
			if okSplit(leftNoNaN, rightWithNaN2, t.MinSamplesLeaf) {
				leftCounts := countsFromIndices(y, leftNoNaN, nClasses, t.classes)
				rightCounts := countsFromIndices(y, rightWithNaN2, nClasses, t.classes)
				impL := impurity(leftCounts)
				impR := impurity(rightCounts)
				weighted := (float64(len(leftNoNaN))/float64(len(idx)))*impL + (float64(len(rightWithNaN2))/float64(len(idx)))*impR
				gain := parentImpurity - weighted
				if gain > result.gain {
					result = splitResult{gain: gain, feature: f, threshold: uv, isCat: true, leftIdx: leftNoNaN, rightIdx: rightWithNaN2}
				}
			}
		}
	}

	// ---- NUMERIC splits: sort valid and scan thresholds ----
	sort.Slice(valid, func(a, b int) bool { return valid[a].v < valid[b].v })

	// move one by one from right to left scanning possible split between distinct values
	for s := 1; s < len(valid); s++ {
		// skip if same value
		if valid[s].v == valid[s-1].v {
			continue
		}

		thr := (valid[s-1].v + valid[s].v) / 2.0

		// Try NaNs on left:
		leftWithNaN := append(append([]int(nil), indicesFromPairs(valid[:s])...), nans...)
		rightNoNaN := indicesFromPairs(valid[s:])
		if okSplit(leftWithNaN, rightNoNaN, t.MinSamplesLeaf) {
			lc := countsFromIndices(y, leftWithNaN, nClasses, t.classes)
			rc := countsFromIndices(y, rightNoNaN, nClasses, t.classes)
			impL := impurity(lc)
			impR := impurity(rc)
			weighted := (float64(len(leftWithNaN))/float64(len(idx)))*impL + (float64(len(rightNoNaN))/float64(len(idx)))*impR
			gain := parentImpurity - weighted
			if gain > result.gain {
				result = splitResult{gain: gain, feature: f, threshold: thr, isCat: false, leftIdx: leftWithNaN, rightIdx: rightNoNaN}
			}
		}
		// Try NaNs on right:
		leftNoNaN := indicesFromPairs(valid[:s])
		rightWithNaN := append(append([]int(nil), indicesFromPairs(valid[s:])...), nans...)
		if okSplit(leftNoNaN, rightWithNaN, t.MinSamplesLeaf) {
			lc := countsFromIndices(y, leftNoNaN, nClasses, t.classes)
			rc := countsFromIndices(y, rightWithNaN, nClasses, t.classes)
			impL := impurity(lc)
			impR := impurity(rc)
			weighted := (float64(len(leftNoNaN))/float64(len(idx)))*impL + (float64(len(rightWithNaN))/float64(len(idx)))*impR
			gain := parentImpurity - weighted
			if gain > result.gain {
				result = splitResult{gain: gain, feature: f, threshold: thr, isCat: false, leftIdx: leftNoNaN, rightIdx: rightWithNaN}
			}
		}
	}
	return result
}

// ---------------------------
// Helpers used in buildNode
// ---------------------------

func almostInt(v float64) bool {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return false
	}
	_, frac := math.Modf(math.Abs(v))
	return frac < 1e-9 || frac > 1-1e-9
}

func uniqueValuesFromPairs(pairs []pair) []float64 {
	m := make(map[float64]struct{})
	out := make([]float64, 0, len(pairs))
	for _, p := range pairs {
		if _, ok := m[p.v]; !ok {
			m[p.v] = struct{}{}
			out = append(out, p.v)
		}
	}
	sort.Float64s(out)
	return out
}

func indicesFromPairs(pairs []pair) []int {
	out := make([]int, 0, len(pairs))
	for _, p := range pairs {
		out = append(out, p.i)
	}
	return out
}

func countsFromIndices(y []int, idx []int, nClasses int, classes []int) []int {
	counts := make([]int, nClasses)
	if len(idx) == 0 {
		return counts
	}
	for _, ii := range idx {
		ci := classIndex(y[ii], classes)
		counts[ci]++
	}
	return counts
}

func okSplit(left, right []int, minLeaf int) bool {
	if len(left) < minLeaf || len(right) < minLeaf {
		return false
	}
	return true
}

// ---------------------------
// Prediction helper
// ---------------------------

func (t *DecisionTreeClassifier) predictProbaSingle(x []float64) []float64 {
	if t.root == nil {
		p := make([]float64, len(t.classes))
		for i := range p {
			p[i] = 1.0 / float64(len(p))
		}
		return p
	}
	node := t.root
	for !node.isLeaf {
		val := x[node.feature]
		if math.IsNaN(val) {
			// missing: choose branch with more samples (heuristic)
			ln := 0
			rn := 0
			if node.left != nil {
				ln = node.left.n
			}
			if node.right != nil {
				rn = node.right.n
			}
			if ln >= rn {
				node = node.left
			} else {
				node = node.right
			}
			continue
		}
		if node.isCat {
			if val == node.threshold {
				node = node.left
			} else {
				node = node.right
			}
		} else {
			if val <= node.threshold {
				node = node.left
			} else {
				node = node.right
			}
		}
	}
	return node.probas
}

// ---------------------------
// Utilities: impurity & misc
// ---------------------------

func giniFromCounts(counts []int) float64 {
	n := 0.0
	for _, c := range counts {
		n += float64(c)
	}
	if n == 0 {
		return 0
	}
	res := 0.0
	for _, c := range counts {
		p := float64(c) / n
		res += p * (1 - p)
	}
	return res
}

func entropyFromCounts(counts []int) float64 {
	n := 0.0
	for _, c := range counts {
		n += float64(c)
	}
	if n == 0 {
		return 0
	}
	res := 0.0
	for _, c := range counts {
		if c == 0 {
			continue
		}
		p := float64(c) / n
		res -= p * math.Log2(p)
	}
	return res
}

func isPure(counts []int) bool {
	nonZero := 0
	for _, c := range counts {
		if c > 0 {
			nonZero++
		}
	}
	return nonZero <= 1
}

func countsToProbas(counts []int) []float64 {
	n := 0
	for _, c := range counts {
		n += c
	}
	p := make([]float64, len(counts))
	if n == 0 {
		return p
	}
	for i := range counts {
		p[i] = float64(counts[i]) / float64(n)
	}
	return p
}

func argmax(counts []int) int {
	best := 0
	for i := 1; i < len(counts); i++ {
		if counts[i] > counts[best] {
			best = i
		}
	}
	return best
}

// classIndex returns index of label in classes slice.
func classIndex(label int, classes []int) int {
	for i, v := range classes {
		if v == label {
			return i
		}
	}
	return 0
}

// ---------------------------
// Reduced-error pruning implementation
// ---------------------------

// pruneNodeReducedError traverses post-order and attempts to prune internal nodes if that increases validation accuracy.
func (t *DecisionTreeClassifier) pruneNodeReducedError(node *dtNode, Xval [][]float64, yval []int, baselineAcc float64) int {
	if node == nil || node.isLeaf {
		return 0
	}
	pruned := 0
	pruned += t.pruneNodeReducedError(node.left, Xval, yval, baselineAcc)
	pruned += t.pruneNodeReducedError(node.right, Xval, yval, baselineAcc)

	if node.left != nil && node.right != nil && node.left.isLeaf && node.right.isLeaf {
		origLeft, origRight, origIsLeaf := node.left, node.right, node.isLeaf
		origFeature, origThresh, origIsCat := node.feature, node.threshold, node.isCat

		nLeft := node.left.n
		nRight := node.right.n
		combinedProbas := make([]float64, len(node.left.probas))
		for i := range combinedProbas {
			combinedProbas[i] = (node.left.probas[i]*float64(nLeft) + node.right.probas[i]*float64(nRight)) / float64(nLeft+nRight)
		}
		node.isLeaf = true
		node.left = nil
		node.right = nil
		node.probas = combinedProbas
		node.predIndex = argmaxFloat(combinedProbas)
		newAcc := accuracyInt(yval, t.Predict(Xval))
		if newAcc >= baselineAcc {
			pruned++
			return pruned
		}
		// revert
		node.left = origLeft
		node.right = origRight
		node.isLeaf = origIsLeaf
		node.feature = origFeature
		node.threshold = origThresh
		node.isCat = origIsCat
		node.probas = nil
		node.predIndex = 0
	}
	return pruned
}

func argmaxFloat(arr []float64) int {
	best := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[best] {
			best = i
		}
	}
	return best
}

func accuracyInt(yTrue []int, yPred []int) float64 {
	if len(yTrue) == 0 || len(yTrue) != len(yPred) {
		return 0.0
	}
	n := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			n++
		}
	}
	return float64(n) / float64(len(yTrue))
}
