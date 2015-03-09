package com.gorbash.neural;

import com.cedarsoftware.util.io.JsonWriter;
import jdk.nashorn.internal.ir.debug.JSONWriter;

import java.io.FileWriter;
import java.util.List;

import static java.util.Arrays.asList;

/**
 * Created by ars032 on 3/9/2015.
 */
public class Functional {

    public static void main(String[] args) throws Exception {

        List<TrainingElement> suite = asList(
                new TrainingElement(asList(0.0, 0.0), asList(0.0)),
                new TrainingElement(asList(0.0, 1.0), asList(1.0)),
                new TrainingElement(asList(1.0, 0.0), asList(1.0)),
                new TrainingElement(asList(1.0, 1.0), asList(0.0)));

        BPNeuralNetwork network = new BPNeuralNetwork.BPPNeuralNetworkBuilder(2, 1).randomizeBias().setHiddenLayers(asList(5,5)).build();
        BPNNTrainer trainer = BPNNTrainer.build(network);
        NeuralNetworkMonitor monitor = new NeuralNetworkMonitor(network);

        TrainingManager manager = new TrainingManager(trainer, 1000);
        double before = monitor.getTotalError(suite);
        manager.runTraining(suite);
        double after = monitor.getTotalError(suite);

        System.out.println(String.format("Before: %s, After: %s", before, after));


        FileWriter writer = new FileWriter("network.json");
        writer.write(JsonWriter.objectToJson(network));
        writer.close();


    }
}
