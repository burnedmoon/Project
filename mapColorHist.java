package src;

import java.io.IOException;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import org.hipi.image.HipiImageHeader;
import org.hipi.image.FloatImage;
import org.hipi.imagebundle.mapreduce.HibInputFormat;

public class mapColorHist extends Configured implements Tool {

	public static class mapColorHistMapper extends
			Mapper<HipiImageHeader, FloatImage, IntWritable, Text> {

		// Convert HIPI FloatImage to OpenCV Mat
		public Mat convertFloatImageToOpenCVMat(FloatImage floatImage) {

			// Get dimensions of image
			int w = floatImage.getWidth();
			int h = floatImage.getHeight();

			// Get pointer to image data
			float[] valData = floatImage.getData();

			// Initialize 3 element array to hold RGB pixel average
			double[] rgb = { 0.0, 0.0, 0.0 };

			Mat mat = new Mat(h, w, CvType.CV_8UC3);

			// Traverse image pixel data in raster-scan order and update running
			// average
			for (int j = 0; j < h; j++) {
				for (int i = 0; i < w; i++) {
					rgb[0] = (double) valData[(j * w + i) * 3 + 0] * 255.0; // R
					rgb[1] = (double) valData[(j * w + i) * 3 + 1] * 255.0; // G
					rgb[2] = (double) valData[(j * w + i) * 3 + 2] * 255.0; // B
					mat.put(j, i, rgb);
				}
			}

			return mat;
		}

		@Override
		public void map(HipiImageHeader header, FloatImage image,
				Context context) throws IOException, InterruptedException {


			String output = null;

			if (header == null) {
				output = "Failed to read image header.";
			} else if (image == null) {
				output = "Failed to decode image data.";
			} else {
				// Load OpenCV native library
				try {
					System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
				} catch (UnsatisfiedLinkError e) {
					System.err.println("Native code library failed to load.\n" + e
							+ Core.NATIVE_LIBRARY_NAME);
					System.exit(1);
				}
				Mat images = this.convertFloatImageToOpenCVMat(image);
				MatOfInt bin = new MatOfInt(8, 12, 3);
				ColorDescriptor cd = new ColorDescriptor(bin);
				List<Mat> feature = cd.describe(images);
				String name = header.getMetaData("filename");
				output = name;
				for (int i = 0; i < 5; i++) {

					for (int j = 0; j < 288; j++) {
						String elementOfMat = "" + feature.get(i).get(j, 0)[0];
						output = output + "," + elementOfMat;
					}
				}

			}

			context.write(new IntWritable(1), new Text(output));
		}

	}

	public static class mapColorHistReducer extends
			Reducer<IntWritable, Text, IntWritable, Text> {

		@Override
		public void reduce(IntWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			for (Text value : values) {
				context.write(key, value);
			}
		}
	}

	public int run(String[] args) throws Exception {
		if (args.length < 2) {
			System.out.println("Usage: getColorHist <input hib> <outputdir>");
			System.exit(0);
		}

		Configuration conf = this.getConf();// new Configuration();

		Job job = Job.getInstance(conf, "hibHsvHist");

		job.setJarByClass(mapColorHist.class);
		job.setMapperClass(mapColorHistMapper.class);
		job.setReducerClass(mapColorHistReducer.class);

		job.setInputFormatClass(HibInputFormat.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);

		String inputPath = args[0];
		String outputPath = args[1];

		removeDir(outputPath, conf);

		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));

		job.setNumReduceTasks(1);

		return job.waitForCompletion(true) ? 0 : 1;

	}

	private static void removeDir(String path, Configuration conf)
			throws IOException {
		Path output_path = new Path(path);
		FileSystem fs = FileSystem.get(conf);
		if (fs.exists(output_path)) {
			fs.delete(output_path, true);
		}
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new mapColorHist(), args);
		System.exit(res);
	}

}
