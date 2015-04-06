import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
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
		String[] args = {"src/train-labels.idx1-ubyte", "src/train-images.idx3-ubyte"};
		try {
			int datasetSize = 10;
			RealMatrix[][] data = importMNIST(args, datasetSize);
			double length = (double) data.length;
			RealMatrix[][] trainingData = new RealMatrix[(int) (length * 0.9)][];
			RealMatrix[][] testData = new RealMatrix[(int) (length * 0.1)][];
			System.arraycopy(data, 0, trainingData, 0, trainingData.length);
			System.arraycopy(data, trainingData.length, testData, 0, testData.length);

			int[] sizes = {784, 30, 10};
			Network net = new Network(sizes);
			
			Wrapper wrapper = loadNet();
			RealMatrix[] weights = wrapper.geWeights();
			RealMatrix[] biases = wrapper.getBiases();
			double oldAccuracy = wrapper.getAccuracy();
			if(weights != null && biases != null) {
				int testNumber = (int)(Math.random() * datasetSize);
				System.out.println("Test of saved weights and biases:");
				RealMatrix expectedOutput = data[testNumber][1];
				System.out.println("Expected Output(" + oldAccuracy + "%): " + expectedOutput.getColumnVector(0).getMaxIndex());
				System.out.println(expectedOutput);
				RealMatrix trainedOutput = net.feedForward(data[testNumber][0], weights, biases);
				System.out.println("Trained Output: " + trainedOutput.getColumnVector(0).getMaxIndex());
				System.out.println(trainedOutput);
			}

			/*double accuracy = net.sgd(trainingData, 1, 50, 3.0, testData);
			System.out.println("Final Accuracy: " + accuracy + "%");
			saveNet(net.weights, net.biases, accuracy);*/
		} catch (IOException | ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private Wrapper loadNet() throws IOException, ClassNotFoundException {
		Wrapper net = null;
		// Read from disk using FileInputStream
		FileInputStream f_in = new FileInputStream("net.data");
		// Read object using ObjectInputStream
		ObjectInputStream obj_in = new ObjectInputStream (f_in);
		// Read an object
		Object obj = obj_in.readObject();
		obj_in.close();
		if (obj instanceof Wrapper)
		{
			net = (Wrapper) obj;
			
		}
		return net;
	}

	private void saveNet(RealMatrix[] weights, RealMatrix[] biases, double accuracy) throws IOException {
		Wrapper net = new Wrapper(weights, biases, accuracy);
		
		// Write to disk with FileOutputStream
		FileOutputStream f_out = new FileOutputStream("net.data");
		// Write object with ObjectOutputStream
		ObjectOutputStream obj_out = new ObjectOutputStream (f_out);
		// Write object out to disk
		obj_out.writeObject (net);
		obj_out.close();
		System.out.println("written to disk");
	}

	public static void displayData(RealMatrix[] example) {
		RealMatrix vectorMatrix = example[0];
		double[][] data = null;
		RealMatrix imatrix = MatrixUtils.createRealMatrix(data);
		RealMatrix solutionMatrix = example[1];
		System.out.println(solutionMatrix.getColumnVector(0).getMaxIndex());
		for(int i = 0; i < imatrix.getRowDimension(); i++) {
			double[] vec = imatrix.getRow(i);
			String vecString = "Row " + i + ": ";
			for(double num : vec) {
				vecString += num + ", ";
			}
			System.out.println(vecString);
		}
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
	
	
	/**
	   * @param args
	   *          args[0]: label file; args[1]: data file.
	   * @throws IOException
	   */
	private RealMatrix[][] importMNIST(String[] args, int number) throws IOException {
		RealMatrix[][] data = new RealMatrix[number][3];

		DataInputStream labels = new DataInputStream(new FileInputStream(args[0]));
		DataInputStream images = new DataInputStream(new FileInputStream(args[1]));
		int magicNumber = labels.readInt();
		if (magicNumber != 2049) {
			System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
			System.exit(0);
		}
		magicNumber = images.readInt();
		if (magicNumber != 2051) {
			System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
			System.exit(0);
		}
		int numLabels = labels.readInt();
		int numImages = images.readInt();
		int numRows = images.readInt();
		int numCols = images.readInt();
		if (numLabels != numImages) {
			System.err.println("Image file and label file do not contain the same number of entries.");
			System.err.println("  Label file contains: " + numLabels);
			System.err.println("  Image file contains: " + numImages);
			System.exit(0);
		}
		
		long start = System.currentTimeMillis();
		int numLabelsRead = 0;
		int numImagesRead = 0;
		int j = 0;
		while (labels.available() > 0 && numLabelsRead < numLabels) {
			byte label = labels.readByte();
			numLabelsRead++;
			//double[][] image = new double[numCols][numRows];
			double[][] vectorImage = new double[numCols * numRows][1];
			int i = 0;
			for (int colIdx = 0; colIdx < numCols; colIdx++) {
				for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
		        	//image[colIdx][rowIdx] = ((double) images.readUnsignedByte());
		        	vectorImage[i][0] = ((double) images.readUnsignedByte()) / 255.0;
		        	i++;
		        }
			}
			numImagesRead++;
		
			// At this point, 'label' and 'image' agree and you can do whatever you like with them.
			//RealMatrix imatrix = MatrixUtils.createRealMatrix(image);
			RealMatrix vectorImatrix = MatrixUtils.createRealMatrix(vectorImage);
			vectorImage = null;
			RealMatrix vectorSolution = vectorizeSolution(label);
			data[j][0] = vectorImatrix;
			data[j][1] = vectorSolution;
		    j++;
		    if(j == number) break;
		}
		if (numLabelsRead % 10 == 0) {
			System.out.print(".");
		}
		if ((numLabelsRead % 800) == 0) {
		    System.out.print(" " + numLabelsRead + " / " + numLabels);
		    long end = System.currentTimeMillis();
		    long elapsed = end - start;
		    long minutes = elapsed / (1000 * 60);
		    long seconds = (elapsed / 1000) - (minutes * 60);
		    System.out.println("  " + minutes + " m " + seconds + " s ");
		}
		
		System.out.println();
		long end = System.currentTimeMillis();
		long elapsed = end - start;
		long minutes = elapsed / (1000 * 60);
		long seconds = (elapsed / 1000) - (minutes * 60);
		System.out.println("Read " + numLabelsRead + " samples in " + minutes + " m " + seconds + " s ");
		return data;
	}
	
	private RealMatrix vectorizeSolution(byte number) {
		RealMatrix matrix = MatrixUtils.createRealMatrix(10, 1);
		matrix.setEntry(number, 0, 1);
		return matrix;
	}
}
