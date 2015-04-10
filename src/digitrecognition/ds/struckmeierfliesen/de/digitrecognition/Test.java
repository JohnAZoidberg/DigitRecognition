package digitrecognition.ds.struckmeierfliesen.de.digitrecognition;

import java.awt.*;
import java.awt.color.*;
import java.awt.image.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.swing.*;

import org.apache.commons.math3.linear.*;


/**
 * Created by Daniel Schäfer on 04.04.2015.
 */

public class Test {
	String[] args = {"train-labels.idx1-ubyte", "train-images.idx3-ubyte"};
	String[] testArgs = {"t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte"};
	Network net;
	
	
	public static void main(String[] args) throws IOException {
		//new Test(null);
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
	}
	public Test() {
		int[] sizes = {784, 30, 10};
		net = new Network(sizes);
		
		try {
			Wrapper wrapper = loadNet("net.data");
			RealMatrix[] weights = wrapper.geWeights();
			RealMatrix[] biases = wrapper.getBiases();
			double oldAccuracy = wrapper.getAccuracy();
			net.setWeightsBiases(weights, biases);			
		} catch (ClassNotFoundException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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
					displayImage(checkImage);
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
					RealMatrix[][] testData = importMNIST(testArgs, 0, 10000);
					System.out.println("Expected Accuracy: " + oldAccuracy + "%");
					//testNet(net, weights, biases, testData);
					int testNumber = (int)(Math.random() * datasetSize);
					image = testData[testNumber];//loadImage("own5.png", 5);
					displayData(image);
					displayImage(image);
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
	
	public int[] guessDigit(RealMatrix[] checkImage, RealMatrix[] weights, RealMatrix[] biases) {
		RealMatrix trainedOutput = net.feedForward(checkImage[0], weights, biases);
		int firstGuess = trainedOutput.getColumnVector(0).getMaxIndex();
		RealVector column = trainedOutput.getColumnVector(0);
		column.setEntry(firstGuess, 0);
		trainedOutput.setColumnVector(0, column);
		int secondGuess = trainedOutput.getColumnVector(0).getMaxIndex();
		int[] guesses = {firstGuess, secondGuess};
		displayImage(checkImage);
		return guesses;
	}
	
	public int[] guessDigit(RealMatrix[] checkImage) {
		RealMatrix[][] weightsBiases = net.getWeightsBiases();
		return guessDigit(checkImage, weightsBiases[0], weightsBiases[1]);
	}
	
	private static void displayImage(RealMatrix[] example) {
		RealMatrix vectorMatrix = example[0];
		int[] buffer = new int[784];
		for(int row = 0; row < 784; row++) {
			buffer[row] = 255 - (int) (vectorMatrix.getEntry(row, 0) * 255);
		}
		
		int VERTICAL_PIXELS = 28;
		int HORIZONTAL_PIXELS = VERTICAL_PIXELS;
		BufferedImage image = getImageFromArray(buffer, VERTICAL_PIXELS, HORIZONTAL_PIXELS);
		
		//image = DrawWindow.resize(image, 280, 280);
		
		ImageUtils.displayImage(image, example[1].getColumnVector(0).getMaxIndex());
	}
	
	private void testNet(Network net, RealMatrix[] weights, RealMatrix[] biases, RealMatrix[][] testData) {
		double accSum = 0.0;
		//for(int i = 0; i < 10000; i+=5000) {
			//RealMatrix[][] testData = importMNIST(testArgs, i, i + 5000);
			int hits = net.evaluate(testData, weights, biases);
			double accuracy = (((double) hits) / testData.length) * 100.0;
			System.out.println(hits + " / " + testData.length + "  " + accuracy + "%");
			accSum += accuracy;
		//}
		System.out.println("Average Accuracy: " + accuracy + "%");
	}

	private static RealMatrix[] loadImage(String path, int number) throws IOException {
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
	}
	
	private Wrapper loadNet(String path) throws IOException, ClassNotFoundException {
		Wrapper net = null;
		// Read from disk using FileInputStream
		InputStream f_in = (InputStream) this.getClass().getResourceAsStream(path);
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
			//System.out.println("Read samples " + startIndex + " to " + endIndex + "(" + (numLabelsRead + 1 - startIndex) + ") in " + minutes + " m " + seconds + " s ");
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
	
	/**
	 * Converts a given Image into a BufferedImage
	 *
	 * @param img The Image to be converted
	 * @return The converted BufferedImage
	 */
	public static BufferedImage toBufferedImage(Image img) {
		if (img instanceof BufferedImage) {
		    return (BufferedImage) img;
		}
		
		// Create a buffered image with transparency
		BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);
		
		// Draw the image on to the buffered image
		Graphics2D bGr = bimage.createGraphics();
		bGr.drawImage(img, 0, 0, null);
		bGr.dispose();
		
		// Return the buffered image
		return bimage;
	}
	
	public static BufferedImage getImageFromArray(int[] pixels, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = (WritableRaster) image.getData();
        raster.setPixels(0, 0, width, height, pixels);
        image.setData(raster);
        return image;
    }
}
