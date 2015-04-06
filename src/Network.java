import java.util.Arrays;
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
        	double[][] weightData = new double[sizes[l + 1]][sizes[l]];
        	for(int j = 0; j < sizes[l]; j++) {
	        	for(int i = 0; i < sizes[l + 1]; i++) {
	        		weightData[i][j] = random.nextGaussian();
	        	}
        	}
        	weights[l] = MatrixUtils.createRealMatrix(weightData);

        	double[][] biasData = new double[1][sizes[l + 1]];
        	for(int i = 0; i < sizes[l + 1]; i++) {
        		biasData[0][i] = random.nextGaussian();
        	}
        	biases[l] = MatrixUtils.createRealMatrix(biasData).transpose();
        }
    }
    
    public void sgd(RealMatrix[][] trainingData, int epochs, int miniBatchSize, double eta, RealMatrix[][] testData) {
    	//int n = trainingData[0].getRowDimension();
    	for(int i = 0; i < epochs; i++) {
    		int trainingSize = trainingData.length;
    		for(int j = 0; j + miniBatchSize < trainingSize; j+=miniBatchSize) {
    			//int end = (1 + j) * miniBatchSize;
        		RealMatrix[][] miniBatch = Arrays.copyOfRange(trainingData, j, j + miniBatchSize);
        		updateMiniBatch(miniBatch, eta);
    		}
    		if(testData != null) {
    			//double accuracy = evaluate(testData) * 100.0;
    			//System.out.println("Epoch " + i + " complete: " + new DecimalFormat("#.00").format(accuracy) + "%");
    			int hits = evaluate(testData);
    			System.out.println("Epoch " + i + ": " + hits + " / " + testData.length + "  " + ((double) hits) / testData.length);
    			System.out.println();
    		}else {
    			System.out.println("Epoch " + i + " complete");
    			System.out.println();
    		}
    	}
    }
    
    public int evaluate(RealMatrix[][] testDatas) {
		double sumOfDifferences = 0.0;
		double sumOfPercentages = 0.0;
		int sumOfMatches = 0;
		int tests = 0;
		boolean whut = true;
		for(RealMatrix[] testData : testDatas) {
			RealMatrix testResult = feedForward(testData[0]);
			RealVector resultVector = testResult.getColumnVector(0);
			int resultInt = resultVector.getMaxIndex();
			int desiredInt = testData[1].getColumnVector(0).getMaxIndex();
			String match = "";
			if(resultInt == desiredInt) {
				sumOfMatches++;
				match = "  -- YES!!";
			}
			//System.out.println("Actual: "  + resultInt + ", Desired: " + desiredInt + match);
			tests++;
			if(whut) {
				//System.out.println(testResult);
				whut = false;
			}
			/*int testRows = testResult.getRowDimension();
			for(int i = 0; i < testRows; i++) {
				double result = testResult.getEntry(i, 0);
				double desired = testData[1].getEntry(i, 0);
				sumOfDifferences += Math.abs(result - desired);
				sumOfPercentages += Math.abs(1 - result / desired);
				tests++;
			}*/
		}
		double sum = (double) sumOfMatches;
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
    		RealMatrix z = weight.multiply(activations);
    		z = z.add(biases[i]);
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
}
