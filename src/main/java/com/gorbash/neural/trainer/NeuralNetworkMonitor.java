package com.gorbash.neural.trainer;

import com.gorbash.neural.network.NeuralNetwork;
import org.apache.log4j.Logger;

import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Created by Gorbash on 2015-03-08.
 */
public class NeuralNetworkMonitor {

    private final static Logger logger = Logger.getLogger(NeuralNetworkMonitor.class);

    private final NeuralNetwork network;

    public NeuralNetworkMonitor(NeuralNetwork network) {
        this.network = checkNotNull(network);
    }

    public double getTotalError(List<TrainingElement> suite) {
        double result = 0;
        for (TrainingElement element : suite) {
            List<Double> response = network.giveResponse(element.getInput());
            logger.info(String.format("Input: %s, output: %s, expectedOutput:%s", element.getInput(), response, element.getExpectedOutput()));
            result += calculateError(response, element.getExpectedOutput());
        }
        return result;
    }

    private double calculateError(List<Double> response, List<Double> expectedOutput) {
        checkArgument(response.size() == expectedOutput.size());
        double result = 0;
        for (int i = 0; i < response.size(); i++) {
            result += Math.abs(expectedOutput.get(i) - response.get(i));
        }
        return result;
    }
}
