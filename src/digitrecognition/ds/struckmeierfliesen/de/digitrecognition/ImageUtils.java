package digitrecognition.ds.struckmeierfliesen.de.digitrecognition;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class ImageUtils {

	public static JFrame displayedFrame = null;
    
    public static BufferedImage resizeRatio(BufferedImage img, int maxW, int maxH) {
    	double oldH = (double) img.getHeight();
    	double oldW = (double) img.getWidth();
    	double ratio = oldH / oldW;
    	double newW = maxW;
    	double newH = maxW * ratio;
    	if(ratio > 1) {
    		newW = maxH / ratio;
    		newH = maxH;
    	}
    	return resize(img, (int) newW, (int) newH);
    }
    
    public static BufferedImage resize(BufferedImage img, int newW, int newH) { 
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return dimg;
    }
    
    public static BufferedImage getCroppedImage(BufferedImage source, double tolerance) {
	   // Get our top-left pixel color as our "baseline" for cropping
	   int baseColor = source.getRGB(0, 0);

	   int width = source.getWidth();
	   int height = source.getHeight();

	   int topY = Integer.MAX_VALUE;
	   int topX = Integer.MAX_VALUE;
	   
	   int bottomY = -1, bottomX = -1;
	   for(int y=0; y<height; y++) {
	      for(int x=0; x<width; x++) {
	         if(colorWithinTolerance(baseColor, source.getRGB(x, y), tolerance)) {
	            if(x < topX) topX = x;
	            if(y < topY) topY = y;
	            if(x > bottomX) bottomX = x;
	            if(y > bottomY) bottomY = y;
	         }
	      }
	   }

	   BufferedImage destination = new BufferedImage((bottomX-topX+1), (bottomY-topY+1), BufferedImage.TYPE_INT_ARGB);

	   destination.getGraphics().drawImage(source, 0, 0, 
	               destination.getWidth(), destination.getHeight(), 
	               topX, topY, bottomX, bottomY, null);

	   return destination;
	}
    
    public static boolean colorWithinTolerance(int a, int b, double tolerance) {
	    int aAlpha  = (int)((a & 0xFF000000) >>> 24);   // Alpha level
	    int aRed    = (int)((a & 0x00FF0000) >>> 16);   // Red level
	    int aGreen  = (int)((a & 0x0000FF00) >>> 8);    // Green level
	    int aBlue   = (int)(a & 0x000000FF);            // Blue level

	    int bAlpha  = (int)((b & 0xFF000000) >>> 24);   // Alpha level
	    int bRed    = (int)((b & 0x00FF0000) >>> 16);   // Red level
	    int bGreen  = (int)((b & 0x0000FF00) >>> 8);    // Green level
	    int bBlue   = (int)(b & 0x000000FF);            // Blue level

	    double distance = Math.sqrt((aAlpha-bAlpha)*(aAlpha-bAlpha) +
	                                (aRed-bRed)*(aRed-bRed) +
	                                (aGreen-bGreen)*(aGreen-bGreen) +
	                                (aBlue-bBlue)*(aBlue-bBlue));

	    // 510.0 is the maximum distance between two colors 
	    // (0,0,0,0 -> 255,255,255,255)
	    double percentAway = distance / 510.0d;     

	    return (percentAway > tolerance);
	}
	
	public static BufferedImage embedWithWhiteBackground(BufferedImage image) {
        BufferedImage square = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = square.createGraphics();
        graphics.setPaint(Color.WHITE);
        graphics.fillRect(0, 0, square.getWidth(), square.getHeight());
        image = ImageUtils.layerImages(square, image);
        return image;
	}
	
	
	// Image1 is bigger and is the background
	public static BufferedImage layerImages(BufferedImage background, BufferedImage foreground) {
		int width1 = background.getWidth();
		int height1 = background.getHeight();
		/* Centers at the Center
		 * double deltaW = (double) (width1 - foreground.getWidth());
		double deltaH = (double) (height1 - foreground.getHeight());
		int paddingLeft = (int) (deltaW / 2.0);
		int paddingTop = (int) (deltaH / 2.0);*/
		// Centers at center of mass
		int[] cogs = centerOfMass(foreground);
		int paddingLeft = (int) (width1 / 2.0) - cogs[0];
		int paddingTop = (int) (height1 / 2.0) - cogs[1];
		//System.out.println("Padding top" + paddingTop + "width: " + foreground.getWidth());
		BufferedImage c = new BufferedImage(width1, height1, BufferedImage.TYPE_INT_ARGB);

		Graphics g = c.getGraphics();
		g.drawImage(background, 0, 0, null);
		g.drawImage(foreground, paddingLeft, paddingTop, null);
		return c;
	}
	
	public static void displayImage(BufferedImage image, int nr, boolean closeOld) {		
		if(closeOld && displayedFrame != null) displayedFrame.dispose();
		displayedFrame = new JFrame("Handwritten " + nr);
		if(nr == -2) displayedFrame.setFocusableWindowState(false);
		displayedFrame.getContentPane().add(new JLabel(new ImageIcon(image)));
		displayedFrame.pack();
		displayedFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		displayedFrame.setSize(image.getWidth() + 10, image.getHeight() + 10);
		displayedFrame.setVisible(true);
	}
	
	public static BufferedImage getImageFromArray(int[] pixels, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = (WritableRaster) image.getData();
        raster.setPixels(0, 0, width, height, pixels);
        image.setData(raster);
        return image;
    }
	
	public static int[] centerOfMass(BufferedImage image) {
		double cogX = 0;
		double cogY = 0;
		double total = 0;
		for(double x = 0; x < image.getWidth(); x++) {
			for(double y = 0; y < image.getHeight(); y++) {
	        	int rgb = image.getRGB((int)x, (int)y);
	        	int r = (rgb >> 16) & 0xFF;
	        	int g = (rgb >> 8) & 0xFF;
	        	int b = (rgb & 0xFF);
				double i = ((double) (r + g + b)) / 3.0;
				cogX = cogX + (i * x); 
				cogY = cogY + (i * y);
				total = total + i;
			}
		}
		cogX = cogX / total;
		cogY = cogY / total;
		/*System.out.println(Math.round(cogX));
		System.out.println(Math.round(cogY));
		System.out.println(image.getWidth() + " " +image.getHeight());*/
		return new int[] {(int) Math.round(cogX), (int) Math.round(cogY)};
	}

	public static void saveExample(char keyChar, BufferedImage image) {
		try {
			File imageFile = new File("customTraining" + keyChar + ".png");
			BufferedImage savedImage = ImageIO.read(imageFile);
	        image = ImageUtils.getCroppedImage(image, 0);
	        image = ImageUtils.resizeRatio(image, 20, 20);
	        image = ImageUtils.embedWithWhiteBackground(image);
			
			image = joinBufferedImage(image, savedImage, keyChar);
			ImageIO.write(image, "png", imageFile);
			displayImage(image, -1, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

    public static BufferedImage joinBufferedImage(BufferedImage dataImage, BufferedImage newImage, int row) {
        //do some calculate first
    	BufferedImage mergedImage = null;
        int width = dataImage.getWidth() + newImage.getWidth();
        int height = dataImage.getHeight();
        //create a new buffer and draw two image into the new image
        mergedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = mergedImage.createGraphics();
        Color oldColor = g2.getColor();
        //fill background
        g2.setPaint(Color.WHITE);
        g2.fillRect(0, 0, width, height);
        //draw image
        g2.setColor(oldColor);
        g2.drawImage(dataImage, null, 0, 0);
        g2.drawImage(newImage, null, dataImage.getWidth(), 0);
        g2.dispose();
        return mergedImage;
    }
    
    public static RealMatrix[][] loadCustomTrainingData() {
    	ArrayList<RealMatrix[]> arrayData = new ArrayList<RealMatrix[]>();
    	for(int i = 0; i < 10; i++) {
    		try {
				BufferedImage bigImage = ImageIO.read(new File("customTraining" + i + ".png"));
				BufferedImage[] images = splitImage(bigImage, 28);
				for(BufferedImage image : images) {
					arrayData.add(bufferedImageToRealMatrix(image, i));
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
    	}
    	RealMatrix[][] data = new RealMatrix[arrayData.size()][2];
    	Collections.shuffle(arrayData);
    	for(int i = 0; i < arrayData.size(); i++) {
    		data[i] = arrayData.get(i);
    	}
    	return data;
    }
    
    public static BufferedImage[] splitImage(BufferedImage image, int colWidth) {
    	int rows = 1; //You should decide the values for rows and cols variables  
        int cols = image.getWidth() / colWidth;  
        int chunks = rows * cols;
  
        int chunkWidth = image.getWidth() / cols; // determines the chunk width and height  
        int chunkHeight = image.getHeight() / rows;  
        int count = 0;  
        BufferedImage images[] = new BufferedImage[chunks]; //Image array to hold image chunks  
        for (int x = 0; x < rows; x++) {  
            for (int y = 0; y < cols; y++) {  
                //Initialize the image array with image chunks  
            	images[count] = new BufferedImage(chunkWidth, chunkHeight, image.getType());  
  
                // draws the image chunk  
                Graphics2D gr = images[count++].createGraphics();  
                gr.drawImage(image, 0, 0, chunkWidth, chunkHeight, chunkWidth * y, chunkHeight * x, chunkWidth * y + chunkWidth, chunkHeight * x + chunkHeight, null);  
                gr.dispose();  
            }  
        }
    	return images;
    }

	public static RealMatrix[] bufferedImageToRealMatrix(BufferedImage image, int number) {
		RealMatrix[] data = new RealMatrix[2];
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
		RealMatrix matrix = MatrixUtils.createRealMatrix(10, 1);
		matrix.setEntry(number, 0, 1);
		data[1] = matrix;
		return data;
	}

	/*public static double[][][] bufferedImageToND4JMatrix(BufferedImage image, int number) {
		double[][][] data = new double[2][][];
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
		data[0] = grayData;
		double[][] matrix = new double[][] {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
		matrix[number][0] = 1;
		data[1] = matrix;
		return data;
	}*/
}
