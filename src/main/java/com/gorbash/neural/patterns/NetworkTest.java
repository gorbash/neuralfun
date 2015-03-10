package com.gorbash.neural.patterns;

import com.cedarsoftware.util.io.JsonReader;
import com.cedarsoftware.util.io.JsonWriter;
import com.gorbash.neural.network.BPNeuralNetwork;
import com.gorbash.neural.trainer.TrainingElement;

import java.io.*;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Gorbash on 2015-03-10.
 */
public class NetworkTest {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader("network.json"));
        String line;
        String json = "";
        while ((line = reader.readLine()) !=null) {
            json += line;
        }
        System.out.println(json);
        BPNeuralNetwork network = (BPNeuralNetwork)JsonReader.jsonToJava(json);

        String fileName = "learning/3_broken.txt";
        List<TrainingElement> trainingElements = new PatternsReader().readElements(Arrays.asList(new File(fileName)));

        List<Double> response = network.giveResponse(trainingElements.get(0).getInput());
        int maxIndex = -1, secondIndex = -1;
        double maxValue = -1.0, secondValue = -1.0;

        for (int i = 0; i < response.size(); i++) {
            Double value = response.get(i);
            if (value > maxValue) {
                maxValue = value;
                maxIndex = i;
            } else if (value > secondValue) {
                secondValue = value;
                secondIndex = i;
            }
        }
        System.out.println(String.format("Largest response neuron %d with value %.3f, second neuron %d with value %.3f, whole response %s", maxIndex+1, maxValue, secondIndex+1, secondValue, response ));

        reader.close();
    }
}
