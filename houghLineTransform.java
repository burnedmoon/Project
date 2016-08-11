package src;

import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class houghLineTransform {
	// global rho theta and votes
	private Map<Double, Double> percentage;
	private Map<Double, Double> distanceRatio;

	// generator
	public houghLineTransform() {
		System.load("/home/cloudera/workspace/MapReduceProject/src/src/libopencv_java2411.so");
	};

	public houghLineTransform(Mat image) {
		System.load("/home/cloudera/workspace/MapReduceProject/src/src/libopencv_java2411.so");
		Mat edge = CannyGaussianBlur(image);
		Mat lines = new Mat();
		Mat linesV = new Mat();
		Imgproc.HoughLines(edge, lines, 1, Math.PI / 180, 1);
		// setLines(lines);
		linesV = countingVote(edge, lines);// C++
		setPercentage(peskPercentage(linesV));// set Hough peak Percentage.
		setDistanceRatio(distanceRatio(linesV));// set Distance Ratio.
	}

	// get & set method
	public Map<Double, Double> getPercentage() {
		return percentage;
	}

	public void setPercentage(Map<Double, Double> percentage) {
		this.percentage = percentage;
	}

	public Map<Double, Double> getDistanceRatio() {
		return distanceRatio;
	}

	public void setDistanceRatio(Map<Double, Double> distanceRatio) {
		this.distanceRatio = distanceRatio;
	}

	// step1: GaussianBlur
	// step2: Canny edge map
	public Mat CannyGaussianBlur(Mat image) {
		Mat gray = new Mat();
		Mat edgeImage = new Mat();
		Mat gray8 = new Mat();

		Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY); // gray image Cv_32FC3
		Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0.5); // Blur image by Gauss Size 3 x 3 std 0.5
		gray.convertTo(gray8, CvType.CV_8U); // Conver CV_32 to CV_8C1
		Imgproc.Canny(gray8, edgeImage, 50, 150, 3, false); // edge map by Canny

		return edgeImage;
	}

	// counting number of vote
	public Mat countingVote(Mat edge, Mat lines) {
		Mat linesTemp = lines.clone();
		Map<String, Integer> linesV = new TreeMap<String, Integer>();
		int vote = 1;
		// count numbers
		while (linesTemp.cols() > 0) {
			Imgproc.HoughLines(edge, linesTemp, 1, Math.PI / 180, vote);
			for (int i = 0; i < linesTemp.cols(); i++) {
				String rhoTheta = "" + (linesTemp.get(0, i)[0]) + ","
						+ (linesTemp.get(0, i)[1]);
				linesV.put(rhoTheta, vote);
			}
			vote += 1;
		}

		// change to Mat
		Mat linesChar = new Mat(lines.rows(), lines.cols(), CvType.CV_32FC3); // rho theta and votes
		for (int i = 0; i < lines.cols(); i++) {
			String rhoTheta = "" + (lines.get(0, i)[0]) + ","
					+ (lines.get(0, i)[1]);
			double[] data = new double[3];
			data[0] = lines.get(0, i)[0];
			data[1] = lines.get(0, i)[1];
			data[2] = linesV.get(rhoTheta); // get vote from linesV
			linesChar.put(0, i, data);
		}

		return linesChar;
	}

	public Map<Double, Double> peskPercentage(Mat linesV) {
		Map<Double, Double> p = new TreeMap<Double, Double>();
		for (int i = 0; i < linesV.cols(); i++) {
			if (p.get(linesV.get(0, i)[1]) != null) {
				p.put(linesV.get(0, i)[1], p.get(linesV.get(0, i)[1]) + 1);
			} else {
				p.put(linesV.get(0, i)[1], (double) 1);
			}
		}
		for (double key : p.keySet()) {
			p.put(key, p.get(key) / (linesV.cols()));
		}

		return p;
	}

	public TreeMap<Double, Double> distanceRatio(Mat linesV) {
		Map<Double, Double> r = new TreeMap<Double, Double>();
		// temp 3 map for rho*rho, rho*vote, vote
		Map<Double, Double> rhoSqrt = new TreeMap<Double, Double>();
		Map<Double, Double> rhoWeight = new TreeMap<Double, Double>();
		Map<Double, Double> weight = new TreeMap<Double, Double>();
		// min theta for paper required.
		double minTheta = 0.0;
		for (int i = 0; i < linesV.cols(); i++) {
			if (minTheta > linesV.get(0, i)[0]) {
				minTheta = linesV.get(0, i)[0];
			}
		}
		for (int i = 0; i < linesV.cols(); i++) {
			double rho = linesV.get(0, i)[0] - minTheta; // rho
			double vote = linesV.get(0, i)[2];// weight
			if (rhoSqrt.get(linesV.get(0, i)[1]) != null) {
				rhoSqrt.put(linesV.get(0, i)[1],
						rhoSqrt.get(linesV.get(0, i)[1]) + rho * rho);
				rhoWeight.put(linesV.get(0, i)[1],
						rhoWeight.get(linesV.get(0, i)[1]) + rho * vote);
				weight.put(linesV.get(0, i)[1], weight.get(linesV.get(0, i)[1])
						+ vote);
			} else {
				rhoSqrt.put(linesV.get(0, i)[1], rho * rho);
				rhoWeight.put(linesV.get(0, i)[1], rho * vote);
				weight.put(linesV.get(0, i)[1], vote);
			}
		}
		for (double key : rhoSqrt.keySet()) {
			double DistanceRatio = Math.sqrt(rhoSqrt.get(key))
					/ (rhoWeight.get(key) / weight.get(key));
			r.put(key, DistanceRatio);
		}

		return (TreeMap<Double, Double>) r;
	}

	// main test
	public static void main(String[] args) throws IOException {
		// image path. it may change by HIPI
		String path = "/home/cloudera/TokyoTower/TokyoTower0001.jpg";
		System.load("/home/cloudera/workspace/MapReduceProject/src/src/libopencv_java2411.so");
		Mat image =  Highgui.imread(path);
		houghLineTransform hlt = new houghLineTransform(image);
		System.out.println("Distance Ratio: ");
		for (double key : hlt.getDistanceRatio().keySet()) {
			System.out.println("" + key + "," + hlt.getDistanceRatio().get(key));

		}
		System.out.println(hlt.getDistanceRatio().size());
	}

}
