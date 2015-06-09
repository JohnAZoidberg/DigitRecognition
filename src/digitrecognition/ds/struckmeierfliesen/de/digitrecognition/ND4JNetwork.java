package digitrecognition.ds.struckmeierfliesen.de.digitrecognition;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.*;
import org.nd4j.linalg.api.activation.Sigmoid;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * Created by Daniel Schäfer on 04.04.2015.
 */

public class ND4JNetwork {
    private int layers;
    private RealMatrix[] realWeights;
    private RealMatrix[] realBiases;
    private INDArray[] weights;
    private INDArray[] biases;
	Sigmoid sigmoid = new Sigmoid();

    public ND4JNetwork(int[] sizes) {
        Nd4j.dtype = DataBuffer.DOUBLE;
        layers = sizes.length;
        weights = new INDArray[layers - 1];
        biases = new INDArray[layers - 1];
		
        Random random = new Random();
        // Initialize Weights
        for(int l = 0; l < layers - 1; l++) {
        	double[][] weightData = new double[sizes[l + 1]][sizes[l]];
        	for(int j = 0; j < sizes[l]; j++) {
	        	for(int i = 0; i < sizes[l + 1]; i++) {
	        		weightData[i][j] = random.nextGaussian();
	        	}
        	}
        	weights[l]= Nd4j.create(weightData);

        	double[][] biasData = new double[sizes[l + 1]][1];
        	for(int i = 0; i < sizes[l + 1]; i++) {
        		biasData[i][0] = random.nextGaussian();
        	}
        	biases[l] = Nd4j.create(biasData);
        }
        int j = 0;
    }
    
    public INDArray[][] getWeightsBiases() {
    	INDArray[][] weightsBiases = {weights, biases};
    	return weightsBiases;
    }
    
    public void setWeightsBiases(INDArray[] weights, INDArray[] biases) {
    	this.weights = weights;
    	this.biases = biases;
    }
    
    /*public double sgd(int trainingSize, int epochs, int miniBatchSize, double eta, INDArray[][] testData) {
    	double accuracy = 0.0;
    	for(int i = 0; i < epochs; i++) {
    		for(int j = 0; j + miniBatchSize <= trainingSize; j+=miniBatchSize) {
    			String[] sources = {"train-labels.idx1-ubyte", "train-images.idx3-ubyte"};
    			INDArray[][] miniBatch = Test.importMNIST(sources, j, j + miniBatchSize - 1);
        		updateMiniBatch(miniBatch, eta);
    		}
    		if(testData != null) {
    			int hits = evaluate(testData, null, null);
    			accuracy = (((double) hits) / testData.length) * 100.0;
    			System.out.println("Epoch " + i + ": " + hits + " / " + testData.length + "  " + accuracy + "%");
    		}else {
    			System.out.println("Epoch " + i + " complete");
    			System.out.println();
    		}
    	}
    	return accuracy;
    }*/
    
    /*public double sgd(INDArray[][] trainingData, int epochs, int miniBatchSize, double eta, INDArray[][] testData) {
    	//int n = trainingData[0].getRowDimension();
    	double accuracy = 0.0;
    	for(int i = 0; i < epochs; i++) {
    		int trainingSize = trainingData.length;
    		trainingData = shuffleArray(trainingData);
    		for(int j = 0; j + miniBatchSize <= trainingSize; j+=miniBatchSize) {
    			INDArray[][] miniBatch = Arrays.copyOfRange(trainingData, j, j + miniBatchSize - 1);
        		updateMiniBatch(miniBatch, eta);
    		}
    		
    		if(testData != null) {
    			int hits = evaluate(testData, null, null);
    			accuracy = (((double) hits) / testData.length) * 100.0;
    			System.out.println("Epoch " + i + ": " + hits + " / " + testData.length + "  " + accuracy + "%");
    		}else {
    			System.out.println("Epoch " + i + " complete");
    			System.out.println();
    		}
    	}
    	return accuracy;
    }*/
    
    public double sgd(double[][][][] trainingData, int epochs, int miniBatchSize, double eta, double[][][][] testData) {
    	//int n = trainingData[0].getRowDimension();
    	double accuracy = 0.0;
    	for(int i = 0; i < epochs; i++) {
    		int trainingSize = trainingData.length;
    		//trainingData = shuffleArray(trainingData);
    		for(int j = 0; j + miniBatchSize <= trainingSize; j+=miniBatchSize) {
    			double[][][][] miniBatch = Arrays.copyOfRange(trainingData, j, j + miniBatchSize - 1);
    			
        		updateMiniBatch(miniBatch, eta);
    		}
    		
    		if(testData != null) {
    			int hits = evaluate(testData, null, null);
    			accuracy = (((double) hits) / testData.length) * 100.0;
    			System.out.println("Epoch " + i + ": " + hits + " / " + testData.length + "  " + accuracy + "%");
    		}else {
    			System.out.println("Epoch " + i + " complete");
    			System.out.println();
    		}
    	}
    	return accuracy;
    }

	public int evaluate(double[][][][] testDatas, INDArray[] ffWeights, INDArray[] ffBiases) {
		int sumOfMatches = 0;
		boolean whut = true;
		for(double[][][] testDataArray : testDatas) {
			INDArray[] testData = convertToINDArray(testDataArray);
			INDArray testResult = feedForward(testData[0], ffWeights, ffBiases);
			int resultInt = getArgMaxVector(testResult);
			int desiredInt = getArgMaxVector(testData[1]);
			if(resultInt == desiredInt) {
				sumOfMatches++;
			}
			//System.out.println("Actual: "  + resultInt + ", Desired: " + desiredInt + match);
			if(whut) {
				//System.out.println(testResult);
				whut = false;
			}
		}
		return sumOfMatches;
	}

	private int getArgMaxVector(INDArray vector) {
		int argMax = 0;
		double value = 0;
		for(int i = 0; i < vector.rows(); i++) {
			double valueI = vector.getDouble(0, i);
			if(valueI > value) {
				value = valueI;
				argMax = i;
			}
		}
		return argMax;
	}

	private void updateMiniBatch(double[][][][] miniBatch, double eta) {
		for(double[][][] dataArray : miniBatch) {
			INDArray[] data = convertToINDArray(dataArray);
			//Test.displayData(data);
			INDArray[][] nabla = backpropagate(data[0], data[1]);
			INDArray[] nablaW = nabla[0];
			INDArray[] nablaB = nabla[1];

    		//System.out.println(nablaW[0].mul(eta / ((double) data.length)));
			
	    	for(int l = 0; l < layers - 1; l++) {
	    		INDArray xW = nablaW[l].mul(eta / ((double) dataArray.length));
	    		weights[l] = weights[l].sub(xW);
	    		INDArray xB = nablaB[l].mul(eta / ((double) dataArray.length));
	    		biases[l] = biases[l].sub(xB);
	    	}
		}
    }
    
    private INDArray[] convertToINDArray(double[][][] dataArray) {
    	INDArray[] data = new INDArray[2];
    	data[0] = Nd4j.create(dataArray[0]);
    	data[1] = Nd4j.create(dataArray[1]);
    	return data;
    }

	private INDArray[][] backpropagate(INDArray activation, INDArray y) {
    	int last = layers - 1 - 1;
    	INDArray inputActivation = activation;
    	
    	INDArray[] activations = new INDArray[layers - 1];
    	INDArray[] zs = new INDArray[layers - 1];
    	// nablaW and nablaB are the gradients of the weights and biases
    	INDArray[] nablaW = new INDArray[layers - 1];
    	INDArray[] nablaB = new INDArray[layers - 1];
    	
    	// Feedforward
    	for(int l = 0; l < layers - 1; l++) {
    		INDArray w = weights[l];
    		INDArray aw = w.mmul(activation);
    		INDArray z = aw.add(biases[l]);
    		zs[l] = z;
    		activation = sigmoidVectorized(z);
    		activations[l] = activation;
    	}
    	
    	// Backward pass
    	INDArray delta = costDerivative(activations[last], y).mul(sigmoidPrimeVectorized(zs[last]));
    	nablaB[last] = delta;
    	nablaW[last] = delta.mmul(activations[last - 1].transpose());
    	for(int l = last - 1; l >= 0; l--) {
    		INDArray z = zs[l];
    		INDArray spv = sigmoidPrimeVectorized(z);
    		delta = weights[l + 1].transpose().mmul(delta).mul(spv);
    		nablaB[l] = delta;
    		INDArray a = inputActivation;
    		if(l > 0) a = activations[l - 1];
    		nablaW[l] = delta.mmul(a.transpose());
    	}
    	
    	INDArray[][] nabla = {nablaW, nablaB}; 
    	return nabla;
    }
    
    public INDArray feedForward(INDArray activations, INDArray[] ffWeights, INDArray[] ffBiases) {

		if(ffWeights == null) ffWeights = weights;
		if(ffBiases == null) ffBiases = biases;
			
    	for(int i = 0; i < layers - 1; i++) {
    		INDArray weight = ffWeights[i];
    		INDArray z = weight.mmul(activations);
    		z = z.add(ffBiases[i]);
    		activations = sigmoidVectorized(z);
    		//int j = 0;
    	}
    	return activations;
    }
    
    private INDArray costDerivative(INDArray outputActivations, INDArray y) {
    	return outputActivations.sub(y);
    }
    
    private INDArray sigmoidVectorized(INDArray x) {
    	return sigmoid(x);
    }
    
    private INDArray sigmoidPrimeVectorized(INDArray x) {
    	return sigmoid.applyDerivative(x);
    }
 
    // Implementing Fisher–Yates shuffle
    public static INDArray[][] shuffleArray(INDArray[][] ar) {    	
		Random rnd = new Random();
		for (int i = ar.length - 1; i > 0; i--) {
			int index = rnd.nextInt(i + 1);
			// Simple swap
			INDArray[] a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
		return ar;
    }
 
    // Implementing Fisher–Yates shuffle
    public static double[][][][] shuffleArray(double[][][][] ar) {    	
		Random rnd = new Random();
		for (int i = ar.length - 1; i > 0; i--) {
			int index = rnd.nextInt(i + 1);
			// Simple swap
			double[][][] a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
		return ar;
    }
}
