import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class TFIDF {

  
    public static class Map1 extends Mapper<LongWritable, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text outKey = new Text();

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String filename = fileSplit.getPath().getName();

            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line);

            
            while (tokenizer.hasMoreTokens()) {
                String term = tokenizer.nextToken();

                outKey.set(term + "@" + filename);
                context.write(outKey, one);

                outKey.set("DOC_LEN@" + filename);
                context.write(outKey, one);
            }
        }
    }

    public static class Reduce1 extends Reducer<Text, IntWritable, Text, IntWritable> {

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable v : values) {
                sum += v.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    public static class Map2 extends Mapper<LongWritable, Text, Text, Text> {
        private Text outKey = new Text();
        private Text outVal = new Text();

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

      
            String[] parts = value.toString().split("\t");
            if (parts.length < 2) return;

            String keyText = parts[0];
            String count = parts[1];

            if (keyText.startsWith("DOC_LEN@")) {
                String doc = keyText.substring("DOC_LEN@".length());
                outKey.set(doc);
                outVal.set("DOC_LEN:" + count);
                context.write(outKey, outVal);
            } else {
                String[] termDoc = keyText.split("@");
                if (termDoc.length < 2) return;
                String term = termDoc[0];
                String doc = termDoc[1];

                outKey.set(doc);
                outVal.set(term + ":" + count);
                context.write(outKey, outVal);
            }
        }
    }

    public static class Reduce2 extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text doc, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            int docLength = 0;  // το |d|
            Map<String, Integer> termCounts = new HashMap<>();

            for (Text v : values) {
                String s = v.toString().trim();

                if (s.startsWith("DOC_LEN:")) {
                    String numStr = s.substring("DOC_LEN:".length()).trim();
                    if (numStr.length() == 0) continue;
                    try {
                        docLength = Integer.parseInt(numStr);
                    } catch (NumberFormatException e) {
                        continue;
                    }
                } else {
                    String[] tc = s.split(":");
                    if (tc.length < 2) continue;

                    String term = tc[0].trim();
                    String numStr = tc[1].trim();
                    if (numStr.length() == 0) continue;

                    try {
                        int c = Integer.parseInt(numStr);
                        termCounts.put(term, c);
                    } catch (NumberFormatException e) {
                        continue;
                    }
                }
            }

            if (docLength == 0) return; 

            for (Map.Entry<String, Integer> e : termCounts.entrySet()) {
                String term = e.getKey();
                int ft = e.getValue();

                double tf = (double) ft / (double) docLength;

                Text outKey = new Text(term + "@" + doc.toString());
                Text outVal = new Text("TF:" + tf);
                context.write(outKey, outVal);
            }
        }
    }

    public static class Map3 extends Mapper<LongWritable, Text, Text, Text> {
        private Text outKey = new Text();
        private Text outVal = new Text();

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] parts = value.toString().split("\t");
            if (parts.length < 2) return;

            String keyText = parts[0]; 
            String tfPart = parts[1]; 

            String[] termDoc = keyText.split("@");
            if (termDoc.length < 2) return;

            String term = termDoc[0];
            String doc = termDoc[1];

            if (!tfPart.startsWith("TF:")) return;
            String tfStr = tfPart.substring(3).trim();
            if (tfStr.length() == 0) return;

            outKey.set(term);
            outVal.set(doc + ":" + tfStr);
            context.write(outKey, outVal);
        }
    }

    public static class Reduce3 extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text term, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            List<String> docTfList = new ArrayList<>();
            int docCount = 0;

            for (Text v : values) {
                String s = v.toString().trim();
                if (s.length() == 0) continue;

                docTfList.add(s);
                docCount++;
            }

            if (docCount == 0) return;

            double N = 3.0;
            double idf = Math.log(N / (double) docCount);

            for (String s : docTfList) {
                String[] dt = s.split(":");
                if (dt.length < 2) continue;

                String doc = dt[0].trim();
                String tfStr = dt[1].trim();

                if (tfStr.length() == 0) continue;

                Text outKey = new Text(doc);
                Text outVal = new Text(term.toString() + ":" + tfStr + ":" + idf);
                context.write(outKey, outVal);
            }
        }
    }

    public static class Map4 extends Mapper<LongWritable, Text, Text, Text> {
        private Text outKey = new Text();
        private Text outVal = new Text();

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] parts = value.toString().split("\t");
            if (parts.length < 2) return;

            String doc = parts[0];
            String data = parts[1];

            outKey.set(doc);
            outVal.set(data);
            context.write(outKey, outVal);
        }
    }

    public static class Reduce4 extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text doc, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            Map<String, Double> tfidfMap = new HashMap<>();

            for (Text v : values) {
                String s = v.toString().trim();
                if (s.length() == 0) continue;

                String[] p = s.split(":");
                if (p.length < 3) continue;

                String term = p[0].trim();
                String tfStr = p[1].trim();
                String idfStr = p[2].trim();

                try {
                    double tf = Double.parseDouble(tfStr);
                    double idf = Double.parseDouble(idfStr);
                    double tfidf = tf * idf;

                    tfidfMap.put(term, tfidf);
                } catch (NumberFormatException e) {
                    continue;
                }
            }

            if (tfidfMap.isEmpty()) return;

            List<Map.Entry<String, Double>> list =
                new ArrayList<>(tfidfMap.entrySet());

            Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
                public int compare(Map.Entry<String, Double> a, Map.Entry<String, Double> b) {
                    return Double.compare(b.getValue(), a.getValue());
                }
            });

            int k = 0;
            for (Map.Entry<String, Double> e : list) {
                if (k >= 5) break;
                Text outVal = new Text(e.getKey() + " -> " + e.getValue());
                context.write(doc, outVal);
                k++;
            }
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length < 5) {
            System.err.println("Usage: TFIDF <input> <out1> <out2> <out3> <out4>");
            System.exit(1);
        }

        Configuration conf = new Configuration();

        conf.set("mapreduce.framework.name", "local");
        conf.set("fs.defaultFS", "file:///");

        Job job1 = new Job(conf, "TFIDF-Job1-ftd-doclen");
        job1.setJarByClass(TFIDF.class);
        job1.setMapperClass(Map1.class);
        job1.setReducerClass(Reduce1.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);
        job1.setInputFormatClass(TextInputFormat.class);
        job1.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path(args[1]));
        if (!job1.waitForCompletion(true)) System.exit(1);

        Job job2 = new Job(conf, "TFIDF-Job2-tf");
        job2.setJarByClass(TFIDF.class);
        job2.setMapperClass(Map2.class);
        job2.setReducerClass(Reduce2.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        job2.setInputFormatClass(TextInputFormat.class);
        job2.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.addInputPath(job2, new Path(args[1]));
        FileOutputFormat.setOutputPath(job2, new Path(args[2]));
        if (!job2.waitForCompletion(true)) System.exit(1);

        Job job3 = new Job(conf, "TFIDF-Job3-idf");
        job3.setJarByClass(TFIDF.class);
        job3.setMapperClass(Map3.class);
        job3.setReducerClass(Reduce3.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);
        job3.setInputFormatClass(TextInputFormat.class);
        job3.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.addInputPath(job3, new Path(args[2]));
        FileOutputFormat.setOutputPath(job3, new Path(args[3]));
        if (!job3.waitForCompletion(true)) System.exit(1);

        Job job4 = new Job(conf, "TFIDF-Job4-top5");
        job4.setJarByClass(TFIDF.class);
        job4.setMapperClass(Map4.class);
        job4.setReducerClass(Reduce4.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(Text.class);
        job4.setInputFormatClass(TextInputFormat.class);
        job4.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.addInputPath(job4, new Path(args[3]));
        FileOutputFormat.setOutputPath(job4, new Path(args[4]));
        if (!job4.waitForCompletion(true)) System.exit(1);
    }
}
