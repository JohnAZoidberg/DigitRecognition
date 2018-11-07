package me.danielschaefer.digitrecognition;

import java.io.Serializable;
import org.apache.commons.math3.linear.RealMatrix;

public class Wrapper implements Serializable {
    RealMatrix[] weights;
    RealMatrix[] biases;
    double accuracy;

    public Wrapper(RealMatrix[] weights, RealMatrix[] biases, double accuracy) {
        this.weights = weights;
        this.biases = biases;
        this.accuracy = accuracy;
    }

    public RealMatrix[] geWeights() {
        return weights;
    }

    public RealMatrix[] getBiases() {
        return biases;
    }

    public double getAccuracy() {
        return accuracy;
    }
}
