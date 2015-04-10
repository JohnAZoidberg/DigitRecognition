package digitrecognition.ds.struckmeierfliesen.de.digitrecognition;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;

import javax.imageio.ImageWriteParam;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class ImageUtils {

    
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

	private static boolean colorWithinTolerance(int a, int b, double tolerance) {
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
        graphics.setPaint (Color.WHITE);
        graphics.fillRect (0, 0, square.getWidth(), square.getHeight());
        image = ImageUtils.layerImages(square, image);
        return image;
	}
	
	
	// Image1 is bigger and is the background
	public static BufferedImage layerImages(BufferedImage image1, BufferedImage image2) {
		int width1 = image1.getWidth();
		int height1 = image1.getHeight();
		double deltaW = (double) (width1 - image2.getWidth());
		double deltaH = (double) (height1 - image2.getHeight());
		int paddingLeft = (int) (deltaW / 2.0);
		int paddingTop = (int) (deltaH / 2.0);
		BufferedImage c = new BufferedImage(width1, height1, BufferedImage.TYPE_INT_ARGB);

		Graphics g = c.getGraphics();
		g.drawImage(image1, 0, 0, null);
		g.drawImage(image2, paddingLeft, paddingTop, null);
		return c;
	}
	
	public static void displayImage(BufferedImage image, int nr) {		
		JFrame frame = new JFrame("Handwritten " + nr);
		frame.getContentPane().add(new JLabel(new ImageIcon(image)));
		frame.pack();
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setSize(400, 400);
		frame.setVisible(true);
	}
}
