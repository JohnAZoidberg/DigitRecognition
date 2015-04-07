import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;

import javax.imageio.ImageIO;

import org.apache.commons.math3.linear.*;

/**
 * Created by Daniel Schäfer on 04.04.2015.
 */

public class Test {

	String[] args = {"src/train-labels.idx1-ubyte", "src/train-images.idx3-ubyte"};
	String[] testArgs = {"src/t10k-labels.idx1-ubyte", "src/t10k-images.idx3-ubyte"};
	
	public static void main(String[] args) {
		new Test();
	}
	
	public Test() {
		try {
			RealMatrix[] image = loadImage();
			image[0] = image[0];
			
			int datasetSize = 1000;
			/*RealMatrix[][] data = importMNIST(args, 0, datasetSize);
			double length = (double) data.length;
			RealMatrix[][] trainingData = new RealMatrix[(int) (length * 0.9)][];
			RealMatrix[][] testData = new RealMatrix[(int) (length * 0.1)][];
			System.arraycopy(data, 0, trainingData, 0, trainingData.length);
			System.arraycopy(data, trainingData.length, testData, 0, testData.length);*/
			//RealMatrix[][] trainingData = importMNIST(args, 0, 900);
			RealMatrix[][] testData = importMNIST(testArgs, 0, 10000);
			
			int[] sizes = {784, 30, 10};
			Network net = new Network(sizes);
			
			Wrapper wrapper = loadNet("src/net.data");
			RealMatrix[] weights = wrapper.geWeights();
			RealMatrix[] biases = wrapper.getBiases();
			double oldAccuracy = wrapper.getAccuracy();
			if(weights != null && biases != null) {
				System.out.println("Expected Accuracy: " + oldAccuracy);
				//testNet(net, weights, biases);
				/*int testNumber = (int)(Math.random() * datasetSize);
				RealMatrix[] img = image;//data[testNumber];
				displayData(img);
				System.out.println("Test of saved weights and biases:");
				RealMatrix expectedOutput = img[1];
				System.out.println("Expected Output(" + oldAccuracy + "%): " + expectedOutput.getColumnVector(0).getMaxIndex());
				System.out.println(expectedOutput);
				RealMatrix trainedOutput = net.feedForward(img[0], weights, biases);
				System.out.println("Trained Output: " + trainedOutput.getColumnVector(0).getMaxIndex());
				System.out.println(trainedOutput);
				
				System.out.println();
				testNumber = (int)(Math.random() * 100);
				img = testData[testNumber];
				displayData(img);
				System.out.println("Test of saved weights and biases:");
				expectedOutput = img[1];
				System.out.println("Expected Output(" + oldAccuracy + "%): " + expectedOutput.getColumnVector(0).getMaxIndex());
				System.out.println(expectedOutput);
				trainedOutput = net.feedForward(img[0], weights, biases);
				System.out.println("Trained Output: " + trainedOutput.getColumnVector(0).getMaxIndex());
				System.out.println(trainedOutput);*/
			}

			double accuracy = net.sgd(1000, 5, 100, 3.0, testData);
			System.out.println("Final Accuracy: " + accuracy + "%");
			if(accuracy > 91.0/*oldAccuracy*/) saveNet(net.weights, net.biases, accuracy);
		} catch (IOException | ClassNotFoundException  e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void testNet(Network net, RealMatrix[] weights, RealMatrix[] biases) {
		double accSum = 0.0;
		for(int i = 0; i < 10000; i+=5000) {
			RealMatrix[][] testData = importMNIST(testArgs, i, i + 5000);
			int hits = net.evaluate(testData, weights, biases);
			double accuracy = (((double) hits) / testData.length) * 100.0;
			System.out.println(hits + " / " + testData.length + "  " + accuracy + "%");
			accSum += accuracy;
		}
		System.out.println("Average Accuracy: " + accSum / 2);
	}

	private RealMatrix[] loadImage() throws IOException {
		RealMatrix[] data = new RealMatrix[3];
		BufferedImage image = ImageIO.read(new File( "src/own7.png" ));
		double[][] grayData = new double[784][1];
		int numCols = image.getWidth();
		int numRows = image.getHeight();
		
		int i = 0;
		for (int colIdx = 0; colIdx < numCols; colIdx++) {
			for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
	        	int rgb = image.getRGB(rowIdx, colIdx);
	        	int r = (rgb >> 16) & 0xFF;
	        	int g = (rgb >> 8) & 0xFF;
	        	int b = (rgb & 0xFF);
	        	grayData[i][0] = (double) (255 - (r + g + b) / 3) / 255.0;
	        	i++;
	        }
		}
		data[0] = MatrixUtils.createRealMatrix(grayData);
		data[1] = vectorizeSolution((byte) 7);
		return data;
	}
	
	private Wrapper loadNet(String path) throws IOException, ClassNotFoundException {
		Wrapper net = null;
		// Read from disk using FileInputStream
		FileInputStream f_in = new FileInputStream(path);
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
		double[][] data = new double[28][28];
		int row = 0;
		for(int i = 0; i < 28; i++) {
			for(int j = 0; j < 28; j++) {
				data[j][i] = vectorMatrix.getEntry(row, 0);
				row++;
			}
		}
		RealMatrix imatrix = MatrixUtils.createRealMatrix(data);
		RealMatrix solutionMatrix = example[1];
		System.out.println(solutionMatrix.getColumnVector(0).getMaxIndex());
		for(int i = 0; i < imatrix.getColumnDimension(); i++) {
			double[] vec = imatrix.getColumn(i);
			String vecString = "Row " + i + ": ";
			for(double num : vec) {
				vecString += Math.round(num * 100.0) / 100.0 + ", ";
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
	public static RealMatrix[][] importMNIST(String[] args, int startIndex, int endIndex) {
		RealMatrix[][] data = new RealMatrix[endIndex - startIndex][3];
		try {
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

				if(j >= startIndex) {
					// At this point, 'label' and 'image' agree and you can do whatever you like with them.
					//RealMatrix imatrix = MatrixUtils.createRealMatrix(image);
					RealMatrix vectorImatrix = MatrixUtils.createRealMatrix(vectorImage);
					vectorImage = null;
					RealMatrix vectorSolution = vectorizeSolution(label);
					data[j - startIndex][0] = vectorImatrix;
					data[j - startIndex][1] = vectorSolution;
				}
			    j++;
			    if(j == endIndex) break;
			}
			if (numLabelsRead % 10 == 0) {
				//System.out.print(".");
			}
			if ((numLabelsRead % 800) == 0) {
			    //System.out.print(" " + numLabelsRead + " / " + numLabels);
			    long end = System.currentTimeMillis();
			    long elapsed = end - start;
			    long minutes = elapsed / (1000 * 60);
			    long seconds = (elapsed / 1000) - (minutes * 60);
			    //System.out.println("  " + minutes + " m " + seconds + " s ");
			}
			
			//System.out.println();
			long end = System.currentTimeMillis();
			long elapsed = end - start;
			long minutes = elapsed / (1000 * 60);
			long seconds = (elapsed / 1000) - (minutes * 60);
			System.out.println("Read samples " + startIndex + " to " + endIndex + "(" + (numLabelsRead + 1 - startIndex) + ") in " + minutes + " m " + seconds + " s ");
		}catch(IOException  e) {
			e.printStackTrace();
		}
		return data;
	}
	
	public static RealMatrix vectorizeSolution(byte number) {
		RealMatrix matrix = MatrixUtils.createRealMatrix(10, 1);
		matrix.setEntry(number, 0, 1);
		return matrix;
	}
}
