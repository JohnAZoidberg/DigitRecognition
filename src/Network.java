import java.util.Random;
import java.text.DecimalFormat;

import org.apache.commons.math3.linear.*;

/**
 * Created by Daniel Schäfer on 04.04.2015.
 */

public class Network {
    int layers;
    RealMatrix[] weights;
    RealMatrix[] biases;

    public Network(int[] sizes) {
        layers = sizes.length;
        biases = new RealMatrix[layers - 1];
        weights = new RealMatrix[layers - 1];
		
        Random random = new Random();
        // Initialize Weights
        for(int l = 0; l < layers - 1; l++) {
        	double[][] weightData = new double[sizes[l]][sizes[l]];
        	for(int j = 0; j < sizes[l]; j++) {
	        	for(int i = 0; i < sizes[l]; i++) {
	        		weightData[i][j] = random.nextGaussian();
	        	}
        	}
        	weights[l] = MatrixUtils.createRealMatrix(weightData);

        	double[][] biasData = new double[1][sizes[l]];
        	for(int i = 0; i < sizes[l]; i++) {
        		biasData[0][i] = random.nextGaussian();
        	}
        	biases[l] = MatrixUtils.createRealMatrix(biasData).transpose();
        }
    }
    
    public void sgd(RealMatrix[][] trainingData, int epochs, int miniBatchSize, double eta, RealMatrix[][] testData) {
    	//int n = trainingData[0].getRowDimension();
    	for(int i = 0; i < epochs; i++) {
    		int trainingSize = trainingData.length;
    		for(RealMatrix[] miniBatch : trainingData) {
        		//RealMatrix[] miniBatch = trainingData;
        		updateMiniBatch(miniBatch, eta);
    		}
    		if(testData != null) {
    			double accuracy = evaluate(testData) * 100.0;
    			System.out.println("Epoch " + i + " complete: " + new DecimalFormat("#.00").format(accuracy) + "%");
    			System.out.println();
    		}else {
    			System.out.println("Epoch " + i + " complete");
    			System.out.println();
    		}
    	}
    }
    
    private double evaluate(RealMatrix[][] testDatas) {
		double sumOfDifferences = 0.0;
		double sumOfPercentages = 0.0;
		int tests = 0;
		boolean whut = true;
		for(RealMatrix[] testData : testDatas) {
			RealMatrix testResult = feedForward(testData[0]);
			if(whut) {
				System.out.println(testResult);
				whut = false;
			}
			int testRows = testResult.getRowDimension();
			for(int i = 0; i < testRows; i++) {
				double result = testResult.getEntry(i, 0);
				double desired = testData[1].getEntry(i, 0);
				sumOfDifferences += Math.abs(result - desired);
				sumOfPercentages += Math.abs(1 - result / desired);
				tests++;
			}
		}
		double sum = sumOfPercentages;
		return sum / ((double) tests);
	}

	private void updateMiniBatch(RealMatrix miniBatch[], double eta) {
    	RealMatrix[][] nabla = backpropagate(miniBatch[0], miniBatch[1]);
    	RealMatrix[] nablaW = nabla[0];
    	RealMatrix[] nablaB = nabla[1];
    	
    	for(int l = 0; l < layers - 1; l++) {
    		weights[l] = weights[l].subtract(nablaW[l].scalarMultiply(eta / miniBatch.length));
    		biases[l] = biases[l].subtract(nablaB[l].scalarMultiply(eta / miniBatch.length));
    	}
    }
    
    private RealMatrix[][] backpropagate(RealMatrix activation, RealMatrix y) {
    	int last = layers - 1 - 1;
    	RealMatrix inputActivation = activation;
    	
    	RealMatrix[] activations = new RealMatrix[layers - 1];
    	RealMatrix[] zs = new RealMatrix[layers - 1];
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
    
    public RealMatrix feedForward(RealMatrix activations) {

		//RealMatrix weight = weights[1];
		//RealMatrix weight2 = weights[2];
    	for(int i = 0; i < layers - 1; i++) {
    		RealMatrix weight = weights[i];
    		RealMatrix z = weight.multiply(activations).add(biases[i]);
    		activations = sigmoidVectorized(z);
    		//int j = 0;
    	}
    	return activations;
    }
    
    private RealMatrix hadamard(RealMatrix x, RealMatrix y) {
    	RealVector xVector = x.getColumnVector(0);
    	RealVector yVector = y.getColumnVector(0);
    	RealVector resultVector = xVector.ebeMultiply(yVector);
    	RealMatrix result = MatrixUtils.createRealMatrix(resultVector.getDimension(), 1);
    	result.setColumnVector(0, resultVector);
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
}
