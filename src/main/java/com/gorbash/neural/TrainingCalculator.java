package com.gorbash.neural;

import java.util.ArrayList;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Created by Gorbash on 2015-03-08.
 */
public class TrainingCalculator {

    private double learningRate;

    public TrainingCalculator(double learningRate) {
        this.learningRate = learningRate;
    }

    List<Double> calculateOutputLayerErrorFactors(List<Double> networkOutputs, List<Double> expectedOutput) {
        checkArgument(networkOutputs.size() == expectedOutput.size());
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < expectedOutput.size(); i++) {
            result.add(expectedOutput.get(i) - networkOutputs.get(i));
        }
        return result;
    }

    List<Double> calculateDeltas(List<Double> responses, List<Double> errorFactors) {
        checkArgument(responses.size() == errorFactors.size());
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < responses.size(); i++) {
            double neuronResponse = responses.get(i);
            result.add(neuronResponse * (1 - neuronResponse) * errorFactors.get(i));
        }
        return result;
    }

    double calculateNewBias(Double oldBias, Double delta) {
        return oldBias + learningRate * delta;
    }

    List<Double> calculateNewWeights(List<Double> stimuli, List<Double> weights, double delta) {
        checkArgument(stimuli.size() == weights.size());
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < weights.size(); i++) {
            Double weight = weights.get(i);
            Double nInput = stimuli.get(i);
            double newWeight = weight + learningRate * delta * nInput;
            result.add(newWeight);
        }
        return result;
    }


}
