import java.util.Random;

import org.apache.commons.math3.linear.*;

/**
 * Created by Daniel Schäfer on 04.04.2015.
 */

public class Test {

	public static void main(String[] args) {
		new Test();
	}
	
	public Test() {        
		int[] sizes = {3, 3, 3};
		Network net = new Network(sizes);
		int nr = 1000;
		RealMatrix[][] trainingData = generateData(nr);
		RealMatrix[][] testData = generateData(nr);
		RealMatrix a1 = trainingData[0][0];
		RealMatrix v1 = trainingData[0][1];
		
		System.out.println("Input Activations:");
		System.out.println(a1);
		System.out.println("Desired Output:");
		System.out.println(v1);
		System.out.println("Untrained Output:");
		System.out.println(net.feedForward(a1));
		System.out.println();
		System.out.println("Training:");
		net.sgd(trainingData, 100, 10, 1.0, testData);
        System.out.println();
		System.out.println("Trained Output:");
		System.out.println(net.feedForward(a1));
        System.out.println();
		System.out.println("Test with simple data:");
        double[][] matrixData = { 
        		{1},
        		{0.8},
        		{0.5}
			};
        RealMatrix testActivations = MatrixUtils.createRealMatrix(matrixData);
        System.out.println(testActivations);
		System.out.println(net.feedForward(testActivations));
	}
	
	private RealMatrix[][] generateData(int number) {
		Random random = new Random();
		RealMatrix[][] trainingData = new RealMatrix[number][2];
		for(int i = 0; i < number; i++) {
			double[][] a1Data = {
					{random.nextDouble()},
					{random.nextDouble()},
					{random.nextDouble()}
			};
			trainingData[i][0] = MatrixUtils.createRealMatrix(a1Data);
			double[][] v1Data = {
					{trainingData[i][0].getEntry(0, 0) / 2},
					{trainingData[i][0].getEntry(1, 0) / 2},
					{trainingData[i][0].getEntry(2, 0) / 2}
			};
			trainingData[i][1] = MatrixUtils.createRealMatrix(v1Data);
		}
		return trainingData;
	}
}
