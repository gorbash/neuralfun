package com.gorbash.neural;

import org.apache.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.stream.Collectors;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Created by Gorbash on 2015-03-06.
 */
public class BPNNTrainer implements Trainer {

    private final static Logger logger = Logger.getLogger(BPNNTrainer.class);

    private final BPNeuralNetwork network;
    private TrainingCalculator calculator;

    private BPNNTrainer(BPNeuralNetwork network, double learningRate) {
        this.network = network;
        this.calculator = new TrainingCalculator(learningRate);
    }

    public static BPNNTrainer build(BPNeuralNetwork network) {
        return new BPNNTrainer(checkNotNull(network), 1.0);
    }

    public void train(TrainingElement element) {

        List<Double> inputs = element.getInput();
        List<Double> expectedOutput = element.getExpectedOutput();

        //TODO: refactor!
        checkArgument(inputs.size() == network.getInputsCount(), "Size of input must be the same as size of network input");
        checkArgument(expectedOutput.size() == network.getOutputsCount(), "Size of output must be the same as size of network output");

        List<Double> networkOutputs = network.giveResponse(inputs);
        List<Double> outputErrorFactors = calculator.calculateOutputLayerErrorFactors(networkOutputs, expectedOutput);

        Stack<List<Double>> deltasStack = new Stack<>();

        Layer previousLayer = network.getOutputLayer();
        List<Double> lastLayerDelta = calculator.calculateDeltas(getResponses(previousLayer), outputErrorFactors);

        deltasStack.push(lastLayerDelta);
        List<Double> previousLayerDeltas = lastLayerDelta;
        int hiddenLayersCount = network.getHiddenLayersCount();
        for (int i = hiddenLayersCount - 1; i >= 0; i--) {
            Layer hiddenLayer = network.getHiddenLayer(i);
            List<Double> hiddenLayerErrorFactors = calculateHiddenLayerErrorFactors(hiddenLayer, previousLayer, previousLayerDeltas);
            previousLayerDeltas = calculator.calculateDeltas(getResponses(hiddenLayer), hiddenLayerErrorFactors);
            deltasStack.push(previousLayerDeltas);
            previousLayer = hiddenLayer;
        }

        for (int i = 0; i < network.getHiddenLayersCount(); i++) {
            Layer hiddenLayer = network.getHiddenLayer(i);
            List<Double> deltas = deltasStack.pop();
            updateFreeParams(hiddenLayer, deltas);
        }

        updateFreeParams(network.getOutputLayer(), deltasStack.pop());
    }

    private List<Double> getResponses(Layer layer) {
        return layer.getNeurons().stream().map(n -> n.getResponse()).collect(Collectors.toList());
    }

    private void updateFreeParams(Layer layer, List<Double> deltas) {
        List<Neuron> neurons = layer.getNeurons();
        for (int i = 0; i < neurons.size(); i++) {
            updateNeuronFreeParam(neurons.get(i), deltas.get(i));
        }
    }

    private void updateNeuronFreeParam(Neuron neuron, Double delta) {
        double newBias = calculator.calculateNewBias(neuron.getBias(), delta);
        List<Double> newWeights = calculator.calculateNewWeights(neuron.getStimuli(), neuron.getWeights(), delta);
        logger.debug(String.format("Updating %s: weights to %s, bias to %s", neuron, newWeights, newBias));
        neuron.setWeights(newWeights);
        neuron.setBias(newBias);
    }


    private List<Double> calculateHiddenLayerErrorFactors(Layer hiddenLayer, Layer previousLayer, List<Double> previousLayerDeltas) {

        //TODO: move to calculator
        List<Double> result = new ArrayList<>();
        List<Neuron> neurons = hiddenLayer.getNeurons();
        for (int i = 0; i < neurons.size(); i++) {
            result.add(calculateErrorFactorForNeuron(previousLayer, previousLayerDeltas, i));
        }
        return result;
    }

    private double calculateErrorFactorForNeuron(Layer previousLayer, List<Double> previousLayerDeltas, int inputIndex) {
        //TODO: move to calculator
        double errorFactor = 0;
        for (int j = 0; j < previousLayerDeltas.size(); j++) {
            Neuron previousNeuron = previousLayer.getNeurons().get(j);
            errorFactor += previousNeuron.getWeights().get(inputIndex) * previousLayerDeltas.get(j);
        }
        return errorFactor;
    }
}
