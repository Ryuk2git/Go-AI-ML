package model

import "math"

func MSE(yTrue, yPred []float64) float64 {
	n := float64(len(yTrue))
	s := 0.0
	for i := range yTrue {
		d := yPred[i] - yTrue[i]
		s += d * d
	}
	return s / n
}

func MAE(yTrue, yPred []float64) float64 {
	n := float64(len(yTrue))
	s := 0.0
	for i := range yTrue {
		d := yPred[i] - yTrue[i]
		if d < 0 {
			d = -d
		}
		s += d
	}
	return s / n
}

func RMSE(yTrue, yPred []float64) float64 { return math.Sqrt(MSE(yTrue, yPred)) }

func R2(yTrue, yPred []float64) float64 {
	m := 0.0
	for _, v := range yTrue {
		m += v
	}
	m /= float64(len(yTrue))
	ssTot := 0.0
	ssRes := 0.0
	for i := range yTrue {
		d := yTrue[i] - m
		ssTot += d * d
		r := yTrue[i] - yPred[i]
		ssRes += r * r
	}
	if ssTot == 0 {
		return 0
	}
	return 1 - ssRes/ssTot
}

func Accuracy(yTrue, yPred []int) float64 {
	c := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			c++
		}
	}
	return float64(c) / float64(len(yTrue))
}

func BinaryPredFromProba(proba []float64, threshold float64) []int {
	out := make([]int, len(proba))
	for i, p := range proba {
		if p >= threshold {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
	return out
}

// Classification metrics (binary, labels 0/1)
func AccuracyInt(yTrue []int, yPred []int) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	c := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			c++
		}
	}
	return float64(c) / float64(len(yTrue))
}

func PrecisionRecallF1(yTrue []int, yPred []int) (prec, rec, f1 float64) {
	tp, fp, fn := 0, 0, 0
	for i := range yTrue {
		if yPred[i] == 1 && yTrue[i] == 1 {
			tp++
		}
		if yPred[i] == 1 && yTrue[i] == 0 {
			fp++
		}
		if yPred[i] == 0 && yTrue[i] == 1 {
			fn++
		}
	}
	if tp+fp > 0 {
		prec = float64(tp) / float64(tp+fp)
	}
	if tp+fn > 0 {
		rec = float64(tp) / float64(tp+fn)
	}
	if prec+rec > 0 {
		f1 = 2 * prec * rec / (prec + rec)
	}
	return
}
