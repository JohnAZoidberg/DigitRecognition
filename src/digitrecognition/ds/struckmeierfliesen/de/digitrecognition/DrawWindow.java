package digitrecognition.ds.struckmeierfliesen.de.digitrecognition;

import java.awt.*;
import java.awt.RenderingHints.Key;
import java.awt.event.*;
import java.awt.image.*;
import java.util.*;

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class DrawWindow {
    private BufferedImage canvasImage;
    
    private JPanel gui;
    
    // Drawing color
    private Color color = Color.BLACK;

    private JLabel imageLabel;
    private JLabel guessLabel;
    
    public final int SIZE = 280;
    public final int DEFAULT_STROKE_SIZE = 30;
    
    private Stroke stroke = new BasicStroke(DEFAULT_STROKE_SIZE, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND, 1.7f);
    private RenderingHints renderingHints;

    
    
    public JComponent getGui() {
        if (gui==null) {
            Map<Key, Object> hintsMap = new HashMap<RenderingHints.Key,Object>();
            hintsMap.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            hintsMap.put(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_ENABLE);
            hintsMap.put(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            renderingHints = new RenderingHints(hintsMap); 

            setImage(new BufferedImage(SIZE,SIZE,BufferedImage.TYPE_INT_RGB));
            gui = new JPanel(new BorderLayout(4,4));
            gui.setBorder(new EmptyBorder(5,3,5,3));

            JPanel imageView = new JPanel(new GridBagLayout());
            imageView.setPreferredSize(new Dimension(480,320));
            imageLabel = new JLabel(new ImageIcon(canvasImage));
            JScrollPane imageScroll = new JScrollPane(imageView);
            imageView.add(imageLabel);
            imageLabel.addMouseMotionListener(new ImageMouseMotionListener());
            imageLabel.addMouseListener(new ImageMouseListener());
            gui.add(imageScroll,BorderLayout.CENTER);

            JToolBar tb = new JToolBar();
            tb.setFloatable(false);

            final SpinnerNumberModel strokeModel = new SpinnerNumberModel(DEFAULT_STROKE_SIZE, 20, 35, 1);
            JSpinner strokeSize = new JSpinner(strokeModel);
            ChangeListener strokeListener = new ChangeListener() {
                @Override
                public void stateChanged(ChangeEvent arg0) {
                    Object o = strokeModel.getValue();
                    Integer i = (Integer)o; 
                    stroke = new BasicStroke(
                            i.intValue(),
                            BasicStroke.CAP_ROUND,
                            BasicStroke.JOIN_ROUND,
                            1.7f);
                }
            };
            strokeSize.addChangeListener(strokeListener);
            strokeSize.setMaximumSize(strokeSize.getPreferredSize());
            JLabel strokeLabel = new JLabel("Stroke");
            strokeLabel.setLabelFor(strokeSize);
            strokeLabel.setDisplayedMnemonic('t');
            tb.add(strokeLabel);
            tb.add(strokeSize);

            tb.addSeparator();

            ActionListener clearListener = new ActionListener() {
                public void actionPerformed(ActionEvent arg0) {
                	guessLabel.setText("");
                	clear(canvasImage);
                }
            };
            JButton clearButton = new JButton("Clear");
            tb.add(clearButton);
            clearButton.addActionListener(clearListener);

            guessLabel = new JLabel("");
            tb.add(guessLabel);
            
            gui.add(tb, BorderLayout.PAGE_START);
            
            clear(canvasImage);
        }

        return gui;
    }

    /** Clears the entire image area by painting it with the current color. */
    public void clear(BufferedImage bi) {
        Graphics2D g = bi.createGraphics();
        g.setRenderingHints(renderingHints);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, bi.getWidth(), bi.getHeight());

        g.dispose();
        imageLabel.repaint();
    }

    public void setImage(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        canvasImage = new BufferedImage(w,h,BufferedImage.TYPE_INT_ARGB);

        Graphics2D g = this.canvasImage.createGraphics();
        g.setRenderingHints(renderingHints);
        g.drawImage(image, 0, 0, gui);
        g.dispose();

        if (gui!=null) {
            gui.invalidate();
        }
    }

    public void draw(Point point) {
        Graphics2D g = this.canvasImage.createGraphics();
        g.setRenderingHints(renderingHints);
        g.setColor(this.color);
        g.setStroke(stroke);
        int n = 0;
        g.drawLine(point.x, point.y, point.x+n, point.y+n);
        g.dispose();
        this.imageLabel.repaint();
    }

    class ImageMouseListener extends MouseAdapter {

        @Override
        public void mousePressed(MouseEvent arg0) {
			if(arg0.getButton() == MouseEvent.BUTTON1) {
				color = Color.BLACK;
			}else if(arg0.getButton() == MouseEvent.BUTTON3) {
				color = Color.WHITE;
			}
			draw(arg0.getPoint());
        }

		@Override
		public void mouseReleased(MouseEvent e) {
			guessImage();
		}
    }

    class ImageMouseMotionListener implements MouseMotionListener {

        @Override
        public void mouseDragged(MouseEvent arg0) {
        	draw(arg0.getPoint());
        }

		@Override
		public void mouseMoved(MouseEvent arg0) {
			// TODO Auto-generated method stub
		}

    }
    
    private void guessImage() {
        BufferedImage image = DrawWindow.this.canvasImage;
        image = ImageUtils.getCroppedImage(image, 0);
        //ImageUtils.displayImage(image, -1);
        image = ImageUtils.resizeRatio(image, 28, 28);
        System.out.println("Height: " + image.getHeight() + ", width: " + image.getWidth());
        image = ImageUtils.embedWithWhiteBackground(image);
        
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
		int number = 0;
		data[0] = MatrixUtils.createRealMatrix(grayData);
		RealMatrix matrix = MatrixUtils.createRealMatrix(10, 1);
		matrix.setEntry(number, 0, 1);
		data[1] = matrix;
		
		Test test = new Test();
		int[] guesses = test.guessDigit(data);
		guessLabel.setText("   Either " + guesses[0] + " or " + guesses[1]);
	}
}
