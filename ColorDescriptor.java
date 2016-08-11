package src;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class ColorDescriptor {
	private MatOfInt bin;

	public ColorDescriptor() {
	};

	public ColorDescriptor(MatOfInt bin) {
		this.bin = bin;
	};

	public MatOfInt getBin() {
		return bin;
	}

	public void setBin(MatOfInt bin) {
		this.bin = bin;
	}

	public Mat histogram(Mat images, Mat mask) {
		// extract a 3D color histogram from the masked region of the
		// image, using the supplied number of bins per channel; then
		// normalize the histogram
		MatOfInt bin = getBin();
		MatOfFloat ranges = new MatOfFloat(0.0f, 180.0f, 0.0f, 256.0f, 0.0f,
				256.0f);
		Mat hist = calcHist3D(images, new MatOfInt(0, 1, 2), mask, bin, ranges);
		Core.normalize(hist, hist);
		Mat cHist = hist.clone();
		hist = cHist.reshape(1);

		return hist;
	}

	public List<Mat> describe(Mat image) {
		// convert the image to the HSV color space and initialize
		// the features used to quantify the image
		Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2HSV);
		List<Mat> features = new ArrayList<Mat>();
		// grab the dimensions and compute the center of the image
		int h = image.height();
		int w = image.width();
		double cY = h * 0.5;
		double cX = w * 0.5;
		// divide the image into four rectangles/segments (top-left,
		// top-right, bottom-right, bottom-left)
		double[][] segments = { { 0.0, cX, 0.0, cY }, { cX, w, 0.0, cY },
				{ cX, w, cY, h }, { 0.0, cX, cY, h } };

		// construct an elliptical mask representing the center of the image
		int axesY = (int) (h * 0.75) / 2;
		int axesX = (int) (w * 0.75) / 2;
		Scalar color255 = new Scalar(255, 255, 255);
		Mat ellipMask = Mat.zeros(new Size(w, h), CvType.CV_8U);
		Core.ellipse(ellipMask, new Point(cX, cY), new Size(axesX, axesY), 0,
				0, 360, color255, -1);

		for (int i = 0; i < 4; i++) {
			double startX = segments[i][0];
			double endX = segments[i][1];
			double startY = segments[i][2];
			double endY = segments[i][3];
			Mat cornerMask = Mat.zeros(new Size(w, h), CvType.CV_8U);
			Core.rectangle(cornerMask, new Point(startX, startY), new Point(
					endX, endY), color255, -1);
			Core.subtract(cornerMask, ellipMask, cornerMask);
			// extract a color histogram from the image, then update the feature
			// vector
			Mat hist = histogram(image, cornerMask);
			features.add(hist);
		}

		// extract a color histogram from the elliptical region and
		// update the feature vector
		Mat hist = histogram(image, ellipMask);
		features.add(hist);

		return features;
	}

	// method for 3D HSV hist in java
	public static Mat calcHist3D(Mat images, MatOfInt channels, Mat mask,
			MatOfInt bin, MatOfFloat ranges) {
		int numberOfHist = (int) (bin.get(0, 0)[0] * bin.get(1, 0)[0] * bin
				.get(2, 0)[0]);
		Mat histM = new Mat(numberOfHist, 1, CvType.CV_32FC1);
		double[][][] hist = new double[(int) bin.get(0, 0)[0]][(int) bin.get(1,
				0)[0]][(int) bin.get(2, 0)[0]];
		double[] level = { ranges.get(1, 0)[0] / bin.get(0, 0)[0],
				ranges.get(3, 0)[0] / bin.get(1, 0)[0],
				ranges.get(5, 0)[0] / bin.get(2, 0)[0] };
//		System.out.println(level[0]+","+level[1]+","+level[2]);
		for (int row = 0; row < images.rows(); row++) {
			for (int col = 0; col < images.cols(); col++) {
				if (mask.get(row, col)[0] == 255.0) {
					int[] area = new int[3];
					for (int i = 0; i < images.channels(); i++) {
						double value = images.get(row, col)[i];
						double rank = Math.floor(value / level[i]);
						area[i] = (int) rank;
					}
					hist[area[0]][area[1]][area[2]] += 1;
				}
			}
		}
		int index = 0;
		for (int i = 0; i < bin.get(0, 0)[0]; i++) {
			for (int j = 0; j < bin.get(1, 0)[0]; j++) {
				for (int k = 0; k < bin.get(2, 0)[0]; k++) {
					double data = hist[i][j][k];
					histM.put(index, 0, data);
					index += 1;
				}
			}
		}
		return histM;
	}

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String path = "simitsu1.jpg";
		Mat images = Highgui.imread(path);
//		System.out.println(images);
//		Imgproc.cvtColor(images, images, Imgproc.COLOR_BGR2HSV);
//		
//		int h = images.height();
//		int w = images.width();
//		double cY = h * 0.5;
//		double cX = w * 0.5;
//		int axesY = (int) (h * 0.75) / 2;
//		int axesX = (int) (w * 0.75) / 2;
//		Scalar color255 = new Scalar(255, 255, 255);
//		Mat ellipMask = Mat.zeros(new Size(w, h), CvType.CV_8U);
//		Core.ellipse(ellipMask, new Point(cX, cY), new Size(axesX, axesY), 0,
//				0, 360, color255, -1);
////		System.out.println(axesX+","+axesY);
//		for(int row = 0; row < images.rows();row++){
//			for(int col =0; col < images.cols();col++){
//				
//				if(ellipMask.get(row, col)[0]==255){
//					System.out.println("("+row+","+col+")");
//				}
//				
//				}
//			}
		
		
//		for(int row = 0; row < images.rows();row++){
//			for(int col =0; col < images.cols();col++){
//				if(images.get(row, col)[0]==103.0 && images.get(row, col)[1] == 51.0 && images.get(row, col)[2] == 251.0){
//					System.out.println(row+","+col);
//				}
////				System.out.println(""+images.get(row, col)[0]+","+images.get(row, col)[1]+","+images.get(row, col)[2]);
//			}
//		}

		
//		MatOfInt bin = new MatOfInt(8, 12, 3);
//		ColorDescriptor cd = new ColorDescriptor(bin);
//		List<Mat> histList = cd.describe(images);
//		for(int i = 0 ; i < 288 ; i++){
//			System.out.println(histList.get(0).get(i, 0)[0]);
//		}
		for(int col=0;col<images.cols();col++){
			for(int i = 0; i < 3 ; i++){
				System.out.print(images.get(0, col)[i]+",");
			}
			System.out.print("("+0+","+col+")");
			System.out.println();
		}
	}

}
