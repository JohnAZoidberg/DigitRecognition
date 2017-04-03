package me.danielschaefer.digitrecognition;

import java.awt.image.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.swing.*;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.linear.*;

import com.google.gson.*;


/**
 * Created by Daniel Schäfer on 04.04.2015.
 */

public class Test {

	public enum Mode {
	    EVAL, TRAIN, DRAW
	}
	// Hier kann man einstellen, ob das Netzwerk trainiert werden soll,
	// ob man zeichen kann und dies dann klassifiziert wird oder ob ein
	// paar handgeschriebene und eingescannte Ziffern klassifiziert werden sollen.
	static final Mode mode = Mode.DRAW;

	static String[] paths = {"train-labels.idx1-ubyte", "train-images.idx3-ubyte"};
	static String[] testPaths = {"t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte"};
	Network net;
	//double oldAccuracy = 0d;

	public Test() {
		try {
			int[] sizes = {784, 30, 10};
			net = new Network(sizes);

			RealMatrix[][] weightsBiases = loadJSONNet();
			RealMatrix[] weights = weightsBiases[0];
			RealMatrix[] biases = weightsBiases[1];
			net.setWeightsBiases(weights, biases);

			String weightJson = jsonize(weights);
			String biasJson = jsonize(biases);
			double[][][] weightDoubles = dejsonize(weightJson);
			weights = new RealMatrix[3];
			for(int i = 0; i < weightDoubles.length; i++) {
				weights[i] = MatrixUtils.createRealMatrix(weightDoubles[i]);
			}
			double[][][] biasDoubles = dejsonize(biasJson);
			biases = new RealMatrix[3];
			for(int i = 0; i < biasDoubles.length; i++) {
				biases[i] = MatrixUtils.createRealMatrix(biasDoubles[i]);
			}
			net.setWeightsBiases(weights, biases);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void main(String[] args) throws IOException {
		Test test;
		switch(mode) {
		case DRAW:
			Runnable r = new Runnable() {
	            @Override
	            public void run() {
	                try {
	                    UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
	                } catch (Exception e) {
	                    // use default
	                }
	                DrawWindow bp = new DrawWindow();

	                JFrame f = new JFrame("Digit Recognition");
	                f.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	                f.setLocationByPlatform(true);

	                f.setContentPane(bp.getGui());

	                f.pack();
	                f.setMinimumSize(f.getSize());
	                f.setVisible(true);
	            }
	        };
	        SwingUtilities.invokeLater(r);
	        break;
		case TRAIN:
			test = new Test();
			test.train();
			break;
		case EVAL:
			test = new Test();
			test.evaluateCustomData();
			break;
		}
	}

	public void evaluateCustomData() {
		RealMatrix[][] trainingData = ImageUtils.loadCustomTrainingData();
		RealMatrix[][] testData = importMNIST(paths, 0, 10000);

		int hits = net.evaluate(testData, null, null);
		double accuracy = (((double) hits) / testData.length) * 100.0;
		System.out.println(hits + " / " + testData.length + "  " + accuracy + "%");

		hits = net.evaluate(trainingData, null, null);
		accuracy = (((double) hits) / trainingData.length) * 100.0;
		System.out.println(hits + " / " + trainingData.length + "  " + accuracy + "%");

		accuracy = net.sgd(trainingData, 10, 10, 0.005, testData);
		System.out.println("Final Accuracy: " + accuracy + "%");

		hits = net.evaluate(trainingData, null, null);
		accuracy = (((double) hits) / trainingData.length) * 100.0;
		System.out.println(hits + " / " + trainingData.length + "  " + accuracy + "%");
		//RealMatrix[][] weightBiases = net.getWeightsBiases();
		//saveJSONNet(weightBiases[0], weightBiases[1]);
	}

	public void train() throws IOException {

		//displayMemoryStats();
		/*int datasetSize = 600;
		RealMatrix[][] data = importMNIST(paths, 0, datasetSize);
		double length = (double) data.length;
		RealMatrix[][] trainingData = new RealMatrix[(int) (length * 0.9)][];
		RealMatrix[][] testData = new RealMatrix[(int) (length * 0.1)][];
		System.arraycopy(data, 0, trainingData, 0, trainingData.length);
		System.arraycopy(data, trainingData.length, testData, 0, testData.length);*/
		RealMatrix[][] trainingData = importMNIST(paths, 0, 5000);
		RealMatrix[][] testData = importMNIST(testPaths, 0, 1000);

		int hits = net.evaluate(testData, null, null);
		double oldAccuracy = (((double) hits) / testData.length) * 100.0;

		int testNumber = (int)(Math.random() * 100);
		RealMatrix[] img = testData[testNumber];
		//displayData(img);
		displayImage(img, false);
		System.out.println("Test of saved weights and biases:");
		RealMatrix expectedOutput = img[1];
		System.out.println("Expected Output(" + oldAccuracy + "%): " + expectedOutput.getColumnVector(0).getMaxIndex());
		RealMatrix trainedOutput = net.feedForward(img[0], null, null);
		System.out.println("Trained Output: " + trainedOutput.getColumnVector(0).getMaxIndex());
		System.out.println(trainedOutput);
		// reset weights and biases to random values
		net.initializeWeightsBiases();
		double accuracy = net.sgd(trainingData, 10, 10, 3.0, testData);
		System.out.println("Final Accuracy: " + accuracy + "%");
		/*if(accuracy > oldAccuracy) {
			RealMatrix[][] weightBiases = net.getWeightsBiases();
			saveNet(weightBiases[0], weightBiases[1], accuracy);
		}*/
	}

	public static String readFile(String path, Charset encoding) throws IOException {
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, encoding);
	}

	public void saveJSONNet(RealMatrix[] weights, RealMatrix[] biases) {
		String weightsJSON = jsonize(weights);
		String biasesJSON = jsonize(biases);
		System.out.println(weightsJSON);
		try {
			FileUtils.writeStringToFile(new File("customWeights.txt"), weightsJSON);
			FileUtils.writeStringToFile(new File("customBiases.txt"), biasesJSON);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static String jsonize(RealMatrix[] matrices) {
		String output = "[";
	    boolean first = true;
	    if(matrices[0] == null) {
	    	System.out.println(0);
	    	return "";
	    }
	    if(matrices[1] == null) {
	    	System.out.println(1);
	    	return "";
	    }
	    for(int i = 0; i < matrices.length; i++) {
	    	RealMatrix layerMatrix = matrices[i];
	    	if(layerMatrix == null) continue;
	    	double[][] layer = layerMatrix.getData();
	        if(first == false)
	        	output += ",";
	        first = true;
	        output += "[";

	        for(double[] row : layer) {
	            if(first == false)
	                output+= ",";
	            first = true;
	            output += "[";

	            for(double element : row) {
	                if(!first)
	                    output += ",";
	                first = false;
	                output += element;
	            }
	            output += "]";
	        }
	        output += "]";
	    }
	    output += "]";
	    return output;
	}

	public static double[][][] dejsonize(String input) {
		Gson gson = new Gson();
		double[][][] data = gson.fromJson(input, double[][][].class);
		return data;
	}

	public Test(RealMatrix[] checkImage) {
		try {
			int datasetSize = 10000;
			/*RealMatrix[][] data = importMNIST(args, 0, datasetSize);
			double length = (double) data.length;
			RealMatrix[][] trainingData = new RealMatrix[(int) (length * 0.9)][];
			RealMatrix[][] testData = new RealMatrix[(int) (length * 0.1)][];
			System.arraycopy(data, 0, trainingData, 0, trainingData.length);
			System.arraycopy(data, trainingData.length, testData, 0, testData.length);*/
			//RealMatrix[][] trainingData = importMNIST(args, 0, 900);

			int[] sizes = {784, 30, 10};
			Network net = new Network(sizes);

			Wrapper wrapper = loadNet("net.data");
			RealMatrix[] weights = wrapper.geWeights();
			RealMatrix[] biases = wrapper.getBiases();
			double oldAccuracy = wrapper.getAccuracy();
			if(weights != null && biases != null) {
				RealMatrix[] image;
				RealMatrix expectedOutput;
				RealMatrix trainedOutput;
				if(checkImage != null) {
					System.out.println("Expected Accuracy: " + oldAccuracy + "%");
					//testNet(net, weights, biases, testData);
					displayData(checkImage);
					displayImage(checkImage, true);
					System.out.println("Test of saved weights and biases:");
					expectedOutput = checkImage[1];
					System.out.println("Expected Output(" + oldAccuracy + "%): " + expectedOutput.getColumnVector(0).getMaxIndex());
					System.out.println(expectedOutput);
					trainedOutput = net.feedForward(checkImage[0], weights, biases);
					int firstGuess = trainedOutput.getColumnVector(0).getMaxIndex();
					System.out.println("Trained Output: " + firstGuess);
					System.out.println(trainedOutput);
					RealVector column = trainedOutput.getColumnVector(0);
					column.setEntry(firstGuess, 0);
					trainedOutput.setColumnVector(0, column);
					int secondGuess = trainedOutput.getColumnVector(0).getMaxIndex();
					JOptionPane.showMessageDialog(null, "First guess: " + firstGuess + ", Second guess: " + secondGuess);
				}else {
					RealMatrix[][] testData = importMNIST(testPaths, 0, 10000);
					System.out.println("Expected Accuracy: " + oldAccuracy + "%");
					//testNet(net, weights, biases, testData);
					int testNumber = (int)(Math.random() * datasetSize);
					image = testData[testNumber];//loadImage("own5.png", 5);
					displayData(image);
					displayImage(image, true);
					System.out.println("Test of saved weights and biases:");
					expectedOutput = image[1];
					System.out.println("Expected Output(" + oldAccuracy + "%): " + expectedOutput.getColumnVector(0).getMaxIndex());
					System.out.println(expectedOutput);
					trainedOutput = net.feedForward(image[0], weights, biases);
					System.out.println("Trained Output: " + trainedOutput.getColumnVector(0).getMaxIndex());
					System.out.println(trainedOutput);
				}

				/*System.out.println();
				testNumber = (int)(Math.random() * 100);
				img = testData[testNumber];
				displayData(img);
				displayImage(img);
				System.out.println("Test of saved weights and biases:");
				expectedOutput = img[1];
				System.out.println("Expected Output(" + oldAccuracy + "%): " + expectedOutput.getColumnVector(0).getMaxIndex());
				System.out.println(expectedOutput);
				trainedOutput = net.feedForward(img[0], weights, biases);
				System.out.println("Trained Output: " + trainedOutput.getColumnVector(0).getMaxIndex());
				System.out.println(trainedOutput);*/
			}

			/*double accuracy = net.sgd(60000, 10, 1000, 3.0, testData);
			System.out.println("Final Accuracy: " + accuracy + "%");
			if(accuracy > oldAccuracy) saveNet(net.weights, net.biases, accuracy);*/
		} catch (IOException | ClassNotFoundException  e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public int[] guessDigit(RealMatrix[] checkImage) {
		RealMatrix trainedOutput = net.feedForward(checkImage[0], null, null);
		//System.out.println(trainedOutput);
		int firstGuess = trainedOutput.getColumnVector(0).getMaxIndex();
		RealVector column = trainedOutput.getColumnVector(0);
		column.setEntry(firstGuess, 0);
		trainedOutput.setColumnVector(0, column);
		int secondGuess = trainedOutput.getColumnVector(0).getMaxIndex();
		int[] guesses = {firstGuess, secondGuess};
		checkImage[1] = vectorizeSolution((byte) firstGuess);
		displayImage(checkImage, true);
		return guesses;
	}

    public int[] guessDigit(BufferedImage image, int number) {
        image = ImageUtils.getCroppedImage(image, 0);
        image = ImageUtils.resizeRatio(image, 20, 20);
        //System.out.println("Height: " + image.getHeight() + ", width: " + image.getWidth());
        image = ImageUtils.embedWithWhiteBackground(image);

        RealMatrix[] data = ImageUtils.bufferedImageToRealMatrix(image, number);

		return guessDigit(data);
	}

	private static void displayImage(RealMatrix[] example, boolean closeOld) {
		RealMatrix vectorMatrix = example[0];
		int[] buffer = new int[784];
		for(int row = 0; row < 784; row++) {
			buffer[row] = 255 - (int) (vectorMatrix.getEntry(row, 0) * 255);
		}

		int VERTICAL_PIXELS = 28;
		int HORIZONTAL_PIXELS = VERTICAL_PIXELS;
		BufferedImage image = ImageUtils.getImageFromArray(buffer, VERTICAL_PIXELS, HORIZONTAL_PIXELS);

		image = ImageUtils.resize(image, 280, 280);

		ImageUtils.displayImage(image, example[1].getColumnVector(0).getMaxIndex(), closeOld);
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

	/*private static RealMatrix[] loadImage(String path, int number) throws IOException {
		RealMatrix[] data = new RealMatrix[3];
		BufferedImage image = ImageIO.read(new File(path));
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
		data[1] = vectorizeSolution((byte) number);
		return data;
	}*/

	private Wrapper loadNet(String path) throws IOException, ClassNotFoundException {
		Wrapper net = null;
		// Read from disk using FileInputStream
		InputStream f_in = new FileInputStream(path);//(InputStream) this.getClass().getResourceAsStream(path);
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

	private RealMatrix[][] loadJSONNet() throws IOException {
		String weightJson = readStringFile("weights.txt");
		String biasJson = readStringFile("biases.txt");

		double[][][] weightDoubles = dejsonize(weightJson);
		RealMatrix[] weights = new RealMatrix[2];
		for(int i = 0; i < weightDoubles.length; i++) {
			weights[i] = MatrixUtils.createRealMatrix(weightDoubles[i]);
		}
		double[][][] biasDoubles = dejsonize(biasJson);
		RealMatrix[] biases = new RealMatrix[2];
		for(int i = 0; i < biasDoubles.length; i++) {
			biases[i] = MatrixUtils.createRealMatrix(biasDoubles[i]);
		}
		return new RealMatrix[][] {weights, biases};
	}

	private String readStringFile(String path) throws IOException {
		FileInputStream fis = new FileInputStream(path);
		BufferedReader in = new BufferedReader(new InputStreamReader(fis));
		String string = org.apache.commons.io.IOUtils.toString(in);
        return string;
	}

	/*private void saveNet(RealMatrix[] weights, RealMatrix[] biases, double accuracy) throws IOException {
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
	}*/

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
			labels.close();
			images.close();

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

	public static double[][] arraycizeSolution(byte number) {
		double[][] array = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
		array[0][number] = 1;
		return array;
	}

	/**
	   * @param args
	   *          args[0]: label file; args[1]: data file.
	   * @throws IOException
	   */
	public static double[][][][] importMNIST(String[] args, int nr) {
		int startIndex = 0;
		int endIndex = nr;
		double[][][][] data = new double[nr][2][][];
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
					//INDArray vectorImatrix = Nd4j.create(vectorImage);
					//INDArray vectorSolution = matricizeSolution(label);
					double[][] arraySolution = arraycizeSolution(label);
					data[j - startIndex][0] = vectorImage;
					data[j - startIndex][1] = arraySolution;
					vectorImage = null;
				}
				if (numLabelsRead % 1000 == 0) {
					System.out.println(numLabelsRead);
				}
				if (numLabelsRead % 5000 == 0) {
					displayMemoryStats();
				}
			    j++;
			    if(j == endIndex) break;
			}
			images.close();
			labels.close();
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
			//System.out.println("Read samples " + startIndex + " to " + endIndex + "(" + (numLabelsRead + 1 - startIndex) + ") in " + minutes + " m " + seconds + " s ");
		}catch(IOException  e) {
			e.printStackTrace();
		}
		return data;
	}

	private static void displayMemoryStats() {
	       int mb = 1024*1024;

	       //Getting the runtime reference from system
	       Runtime runtime = Runtime.getRuntime();

	       System.out.println("##### Heap utilization statistics [MB] #####");

	       //Print used memory
	       System.out.println("Used Memory:"
	           + (runtime.totalMemory() - runtime.freeMemory()) / mb);

	       //Print free memory
	       System.out.println("Free Memory:"
	           + runtime.freeMemory() / mb);

	       //Print total available memory
	       System.out.println("Total Memory:" + runtime.totalMemory() / mb);

	       //Print Maximum available memory
	       System.out.println("Max Memory:" + runtime.maxMemory() / mb);
	}
}
