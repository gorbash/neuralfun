package com.gorbash.neural.trainer;

import com.gorbash.neural.network.BPNeuralNetwork;
import com.gorbash.neural.network.Layer;
import com.gorbash.neural.network.Neuron;
import org.apache.log4j.Logger;

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
    private static final double DEFAULT_LEARNING_RATE = 1.0;

    private final BPNeuralNetwork network;
    private final TrainingCalculator calculator;

    private BPNNTrainer(BPNeuralNetwork network, double learningRate) {
        this.network = network;
        this.calculator = new TrainingCalculator(learningRate);
    }

    public static BPNNTrainer build(BPNeuralNetwork network) {
        return build(checkNotNull(network), DEFAULT_LEARNING_RATE);
    }

    public static BPNNTrainer build(BPNeuralNetwork network, double learningRate) {
        return new BPNNTrainer(checkNotNull(network), learningRate);
    }

    public void train(TrainingElement element) {
        //TODO: refactor!
        validateTrainingElement(element);

        List<Double> networkOutputs = network.giveResponse(element.getInput());
        List<Double> outputErrorFactors = calculator.calculateOutputLayerErrorFactors(networkOutputs, element.getExpectedOutput());

        Layer previousLayer = network.getOutputLayer();
        List<Double> lastLayerDelta = calculator.calculateDeltas(getResponses(previousLayer), outputErrorFactors);

        Stack<List<Double>> deltasStack = new Stack<>();
        deltasStack.push(lastLayerDelta);
        List<Double> previousLayerDeltas = lastLayerDelta;
        int hiddenLayersCount = network.getHiddenLayersCount();
        for (int i = hiddenLayersCount - 1; i >= 0; i--) {
            Layer hiddenLayer = network.getHiddenLayer(i);
            List<Double> hiddenLayerErrorFactors = calculator.calculateHiddenLayerErrorFactors(hiddenLayer, previousLayer, previousLayerDeltas);
            previousLayerDeltas = calculator.calculateDeltas(getResponses(hiddenLayer), hiddenLayerErrorFactors);
            deltasStack.push(previousLayerDeltas);
            previousLayer = hiddenLayer;
        }

        updateFreeParamsOfNetwork(deltasStack);
    }

    private void updateFreeParamsOfNetwork(Stack<List<Double>> deltasStack) {
        for (int i = 0; i < network.getHiddenLayersCount(); i++) {
            Layer hiddenLayer = network.getHiddenLayer(i);
            updateFreeParamsOfLayer(hiddenLayer, deltasStack.pop());
        }
        updateFreeParamsOfLayer(network.getOutputLayer(), deltasStack.pop());
    }

    private void validateTrainingElement(TrainingElement element) {
        checkArgument(element.getInput().size() == network.getInputsCount(), "Size of input must be the same as size of network input");
        checkArgument(element.getExpectedOutput().size() == network.getOutputsCount(), "Size of output must be the same as size of network output");
    }

    private List<Double> getResponses(Layer layer) {
        return layer.getNeurons().stream().map(Neuron::getResponse).collect(Collectors.toList());
    }

    private void updateFreeParamsOfLayer(Layer layer, List<Double> deltas) {
        List<Neuron> neurons = layer.getNeurons();
        for (int i = 0; i < neurons.size(); i++) {
            updateFreeParamsOfNeuron(neurons.get(i), deltas.get(i));
        }
    }

    private void updateFreeParamsOfNeuron(Neuron neuron, Double delta) {
        double newBias = calculator.calculateNewBias(neuron.getBias(), delta);
        List<Double> newWeights = calculator.calculateNewWeights(neuron.getStimuli(), neuron.getWeights(), delta);
        logger.debug(String.format("Updating %s: weights to %s, bias to %s", neuron, newWeights, newBias));
        neuron.setWeights(newWeights);
        neuron.setBias(newBias);
    }


}
