package me.danielschaefer.digitrecognition;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.*;

/**
 * Created by Daniel Schäfer on 04.04.2015.
 */

public class Network {
    private int layers;
    private RealMatrix[] weights;
    private RealMatrix[] biases;
    int[] sizes;

    public Network(int[] sizes) {
        layers = sizes.length;
        biases = new RealMatrix[layers - 1];
        weights = new RealMatrix[layers - 1];

        // Initialize Weights and Biases
        this.sizes = sizes;
        initializeWeightsBiases();
    }

    public void initializeWeightsBiases() {
        Random random = new Random();
        for (int l = 0; l < layers - 1; l++) {
            double[][] weightData = new double[sizes[l + 1]][sizes[l]];
            for (int j = 0; j < sizes[l]; j++) {
                for (int i = 0; i < sizes[l + 1]; i++) {
                    weightData[i][j] = random.nextGaussian();
                }
            }
            weights[l] = MatrixUtils.createRealMatrix(weightData);

            double[][] biasData = new double[sizes[l + 1]][1];
            for (int i = 0; i < sizes[l + 1]; i++) {
                biasData[i][0] = random.nextGaussian();
            }
            biases[l] = MatrixUtils.createRealMatrix(biasData);
        }
    }

    public RealMatrix[][] getWeightsBiases() {
        RealMatrix[][] weightsBiases = {weights, biases};
        return weightsBiases;
    }

    public void setWeightsBiases(RealMatrix[] weights, RealMatrix[] biases) {
        this.weights = weights;
        this.biases = biases;
    }

    public double sgd(int trainingSize, int epochs, int miniBatchSize, double eta, RealMatrix[][] testData) {
        double accuracy = 0.0;
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j + miniBatchSize <= trainingSize; j+=miniBatchSize) {
                String[] sources = {"train-labels.idx1-ubyte", "train-images.idx3-ubyte"};
                RealMatrix[][] miniBatch = Test.importMNIST(sources, j, j + miniBatchSize - 1);
                updateMiniBatch(miniBatch, eta);
            }

            if(testData != null) {
                int hits = evaluate(testData, null, null);
                accuracy = (((double) hits) / testData.length) * 100.0;
                System.out.println("Epoch " + i + ": " + hits + " / " + testData.length + "  " + accuracy + "%");
            } else {
                System.out.println("Epoch " + i + " complete");
                System.out.println();
            }
        }
        return accuracy;
    }

    public double sgd(RealMatrix[][] trainingData, int epochs, int miniBatchSize, double eta, RealMatrix[][] testData) {
        double accuracy = 0.0;
        for(int i = 0; i < epochs; i++) {
            int trainingSize = trainingData.length;
            trainingData = shuffleArray(trainingData);
            for (int j = 0; j + miniBatchSize <= trainingSize; j+=miniBatchSize) {
                RealMatrix[][] miniBatch = Arrays.copyOfRange(trainingData, j, j + miniBatchSize - 1);
                updateMiniBatch(miniBatch, eta);
            }

            if (testData != null) {
                int hits = evaluate(testData, null, null);
                accuracy = (((double) hits) / testData.length) * 100.0;
                System.out.println("Epoch " + i + ": " + hits + " / " + testData.length + "  " + accuracy + "%");
            } else {
                System.out.println("Epoch " + i + " complete");
                System.out.println();
            }
        }
        return accuracy;
    }

    public int evaluate(RealMatrix[][] testDatas, RealMatrix[] ffWeights, RealMatrix[] ffBiases) {
        int sumOfMatches = 0;
        for(RealMatrix[] testData : testDatas) {
            RealMatrix testResult = feedForward(testData[0], ffWeights, ffBiases);
            RealVector resultVector = testResult.getColumnVector(0);
            int resultInt = resultVector.getMaxIndex();
            int desiredInt = testData[1].getColumnVector(0).getMaxIndex();
            if(resultInt == desiredInt) {
                sumOfMatches++;
            }
        }
        return sumOfMatches;
    }

    private void updateMiniBatch(RealMatrix[][] miniBatch, double eta) {
        for(RealMatrix[] data : miniBatch) {
            //Test.displayData(data);
            RealMatrix[][] nabla = backpropagate(data[0], data[1]);
            RealMatrix[] nablaW = nabla[0];
            RealMatrix[] nablaB = nabla[1];

            for(int l = 0; l < layers - 1; l++) {
                weights[l] = weights[l].subtract(nablaW[l].scalarMultiply(eta / ((double) data.length)));
                biases[l] = biases[l].subtract(nablaB[l].scalarMultiply(eta / ((double) data.length)));
            }
        }
    }

    private RealMatrix[][] backpropagate(RealMatrix activation, RealMatrix y) {
        int last = layers - 1 - 1;
        RealMatrix inputActivation = activation;

        RealMatrix[] activations = new RealMatrix[layers - 1];
        RealMatrix[] zs = new RealMatrix[layers - 1];
        // nablaW and nablaB are the gradients of the weights and biases
        RealMatrix[] nablaW = new RealMatrix[layers - 1];
        RealMatrix[] nablaB = new RealMatrix[layers - 1];

        // Feedforward
        for(int l = 0; l < layers - 1; l++) {
            RealMatrix w = weights[l];
            RealMatrix aw = w.multiply(activation);
            RealMatrix z = aw.add(biases[l]);
            zs[l] = z;
            activation = sigmoidVectorized(z);
            activations[l] = activation;
        }

        // Backward pass
        RealMatrix delta = hadamard(costDerivative(activations[last], y), sigmoidPrimeVectorized(zs[last]));
        nablaB[last] = delta;
        nablaW[last] = delta.multiply(activations[last - 1].transpose());
        for(int l = last - 1; l >= 0; l--) {
            RealMatrix z = zs[l];
            RealMatrix spv = sigmoidPrimeVectorized(z);
            delta = hadamard(weights[l + 1].transpose().multiply(delta), spv);
            nablaB[l] = delta;
            RealMatrix a = inputActivation;
            if(l > 0) a = activations[l - 1];
            nablaW[l] = delta.multiply(a.transpose());
        }

        RealMatrix[][] nabla = {nablaW, nablaB};
        return nabla;
    }

    public RealMatrix feedForward(RealMatrix activations, RealMatrix[] ffWeights, RealMatrix[] ffBiases) {

        if(ffWeights == null) ffWeights = weights;
        if(ffBiases == null) ffBiases = biases;

        for(int i = 0; i < layers - 1; i++) {
            RealMatrix weight = ffWeights[i];
            RealMatrix z = weight.multiply(activations);
            z = z.add(ffBiases[i]);
            activations = sigmoidVectorized(z);
            //int j = 0;
        }
        return activations;
    }

    private RealMatrix hadamard(RealMatrix x, RealMatrix y) {
        if(x.getRowDimension() != y.getRowDimension() || x.getColumnDimension() != y.getColumnDimension()) ;
        int columns = x.getColumnDimension();
        int rows = x.getRowDimension();
        RealMatrix result = MatrixUtils.createRealMatrix(rows, columns);
        for(int i = 0; i < columns; i++) {
            RealVector xVector = x.getColumnVector(i);
            RealVector yVector = y.getColumnVector(i);
            RealVector resultVector = xVector.ebeMultiply(yVector);
            result.setColumnVector(i, resultVector);
        }
        return result;
    }

    private RealMatrix costDerivative(RealMatrix outputActivations, RealMatrix y) {
        return outputActivations.subtract(y);
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private RealMatrix sigmoidVectorized(RealMatrix x) {
        double[][] array = x.getData();
        for(int i = 0; i < array.length; i++) {
            array[i][0] = sigmoid(array[i][0]);
        }
        return MatrixUtils.createRealMatrix(array);
    }

    private double sigmoidPrime(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    private RealMatrix sigmoidPrimeVectorized(RealMatrix x) {
        double[][] array = x.getData();
        for(int i = 0; i < array.length; i++) {
            array[i][0] = sigmoidPrime(array[i][0]);
        }
        return MatrixUtils.createRealMatrix(array);
    }

    // Implementing Fisher-Yates shuffle
    public static RealMatrix[][] shuffleArray(RealMatrix[][] ar) {
        Random rnd = new Random();
        for (int i = ar.length - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            RealMatrix[] a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
        return ar;
    }
}
