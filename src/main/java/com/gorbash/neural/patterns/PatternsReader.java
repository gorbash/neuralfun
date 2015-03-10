package com.gorbash.neural.patterns;

import com.cedarsoftware.util.io.JsonWriter;
import com.gorbash.neural.network.BPNeuralNetwork;
import com.gorbash.neural.trainer.BPNNTrainer;
import com.gorbash.neural.trainer.NeuralNetworkMonitor;
import com.gorbash.neural.trainer.TrainingElement;
import com.gorbash.neural.trainer.TrainingManager;

import java.io.*;
import java.util.*;

import static java.util.Arrays.asList;

/**
 * Created by Gorbash on 2015-03-10.
 */
public class PatternsReader {

    private Map<Integer, String> patterns = new HashMap<>();

    public List<TrainingElement> readElements(List<File> files) throws IOException {
        List<TrainingElement> result = new ArrayList<>();
        files.stream().forEach((file1) -> readElement(file1));
        for (File file:files) {
            readElement(file);
        }
        System.out.println(patterns);

        int maxKey = 0;
        for (Integer key:patterns.keySet()) {
            if (key > maxKey)
                maxKey = key;
        }

        System.out.println("max output is " + maxKey);

        for (Map.Entry entry:patterns.entrySet()) {
            Integer neuronNumber = (Integer)entry.getKey();
            String pattern = (String)entry.getValue();
            TrainingElement element = new TrainingElement(getInput(pattern), getExpectedOutput(neuronNumber, maxKey));
            result.add(element);
        }

        return result;
    }

    private List<Double> getExpectedOutput(Integer neuronNumber, int maxKey) {
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < maxKey; i++) {
            if (i == neuronNumber-1)
                result.add(1.0);
            else
                result.add(0.0);
        }
        return result;

    }

    private List<Double> getInput(String pattern) {
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < pattern.length();i++) {
            if (pattern.charAt(i) == '.') {
                result.add(0.0);
            } else if (pattern.charAt(i) == 'x') {
                result.add(1.0);
            } else {
                throw new IllegalArgumentException("Unknown character: " + pattern.charAt(i));
            }
        }
        return result;
    }

    private void readElement(File file){
        FileReader reader = null;
        try {
            reader = new FileReader(file);
            BufferedReader buffReader = new BufferedReader(reader);
            String neuronNumber = buffReader.readLine();
            String pattern = "";
            String line;
            while((line = buffReader.readLine()) != null) {
                pattern += line.trim();
            }
            patterns.put(Integer.parseInt(neuronNumber), pattern);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (reader != null)
                    reader.close();
            }catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void main(String[] args) throws IOException {
        List<TrainingElement> suite = new PatternsReader().readElements(
                Arrays.asList(
                        new File("learning/0.txt"),
                        new File("learning/1.txt"),
                        new File("learning/2.txt"),
                        new File("learning/3.txt"),
                        new File("learning/4.txt"),
                        new File("learning/5.txt"),
                        new File("learning/6.txt"),
                        new File("learning/7.txt"),
                        new File("learning/8.txt"),
                        new File("learning/9.txt")
                ));
        System.out.println(suite);

        BPNeuralNetwork network = new BPNeuralNetwork.BPPNeuralNetworkBuilder(20, 10).randomizeBias().setHiddenLayers(asList(5)).build();
        BPNNTrainer trainer = BPNNTrainer.build(network, 1.0);
        NeuralNetworkMonitor monitor = new NeuralNetworkMonitor(network);

        TrainingManager manager = new TrainingManager(trainer, 10000);
        double before = monitor.getTotalError(suite);
        manager.runTraining(suite);
        double after = monitor.getTotalError(suite);

        System.out.println(String.format("Before: %s, After: %s", before, after));
        FileWriter writer = new FileWriter("network.json");
        writer.write(JsonWriter.objectToJson(network));
        writer.close();
        System.out.println("Network saved to network.json");
    }
}
