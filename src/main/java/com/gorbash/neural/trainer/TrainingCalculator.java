package com.gorbash.neural.trainer;

import com.gorbash.neural.network.Layer;
import com.gorbash.neural.network.Neuron;

import java.util.ArrayList;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Created by Gorbash on 2015-03-08.
 */
public class TrainingCalculator {

    private final double learningRate;

    TrainingCalculator(double learningRate) {
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

    List<Double> calculateNewWeights(List<Double> stimuli, List<Double> weights, double delta) {
        checkArgument(stimuli.size() == weights.size());
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < weights.size(); i++) {
            result.add(calculateNewWeight(stimuli.get(i), weights.get(i), delta, i));
        }
        return result;
    }

    private double calculateNewWeight(double input, double weight, double delta, int i) {
        return weight + learningRate * delta * input;
    }

    double calculateNewBias(Double oldBias, Double delta) {
        return oldBias + learningRate * delta;
    }

    List<Double> calculateHiddenLayerErrorFactors(Layer hiddenLayer, Layer previousLayer, List<Double> previousLayerDeltas) {

        List<Double> result = new ArrayList<>();
        List<Neuron> neurons = hiddenLayer.getNeurons();
        for (int i = 0; i < neurons.size(); i++) {
            result.add(calculateErrorFactorForNeuron(previousLayer, previousLayerDeltas, i));
        }
        return result;
    }

    private double calculateErrorFactorForNeuron(Layer previousLayer, List<Double> previousLayerDeltas, int inputIndex) {

        double errorFactor = 0;
        for (int j = 0; j < previousLayerDeltas.size(); j++) {
            Neuron previousNeuron = previousLayer.getNeurons().get(j);
            errorFactor += previousNeuron.getWeights().get(inputIndex) * previousLayerDeltas.get(j);
        }
        return errorFactor;
    }
}
