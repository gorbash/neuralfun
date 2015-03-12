package com.gorbash.neural.network;

import com.gorbash.neural.tfunc.SigmoidTransfer;
import com.gorbash.neural.tfunc.TransferFunction;
import org.apache.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Created by Gorbash on 2015-02-18.
 * Back Propagation Neural Network
 */
public class BPNeuralNetwork implements NeuralNetwork {

    private final static Logger logger = Logger.getLogger(BPNeuralNetwork.class);

    private List<Layer> layers;
    private final int inputsCount;
    private final int outputsCount;

    private BPNeuralNetwork(int inputsCount, int outputsCount) {
        this.inputsCount = inputsCount;
        this.outputsCount = outputsCount;
    }


    public int getHiddenLayersCount() {
        return layers.size() - 1;
    }

    public int getInputsCount() {
        return inputsCount;
    }

    public int getOutputsCount() {
        return outputsCount;
    }

    public int getNeuronsCountInLayer(int layerIndex) {
        return layers.get(layerIndex).getNeuronsCount();
    }

    public static class BPPNeuralNetworkBuilder {

        private final int inputsCount;
        private final int outputsCount;
        private double bias;
        private TransferFunction function = new SigmoidTransfer();
        private boolean defaultWeightsValue;
        private boolean randomizedBias;
        private List<Integer> neuronCounts;

        public BPPNeuralNetworkBuilder(int inputsCount, int outputsCount) {
            checkArgument(inputsCount >= 1, "At least one input is required!");
            checkArgument(outputsCount >= 1, "At least one output is required");
            this.inputsCount = inputsCount;
            this.outputsCount = outputsCount;
            this.neuronCounts = Collections.emptyList();
        }

        public BPPNeuralNetworkBuilder setTransferFunction(TransferFunction f) {
            this.function = checkNotNull(f);
            return this;
        }

        public BPPNeuralNetworkBuilder defaultWeightsValue() {
            this.defaultWeightsValue = true;
            return this;
        }

        public BPPNeuralNetworkBuilder setBias(double bias) {
            this.bias = bias;
            return this;
        }

        public BPPNeuralNetworkBuilder randomizeBias() {
            this.randomizedBias = true;
            return this;
        }


        public BPPNeuralNetworkBuilder setHiddenLayers(List<Integer> neuronCounts) {
            checkNotNull(neuronCounts);
            checkArgument(neuronCounts.stream().allMatch(p -> p > 0), "All elements must be greater than 0.");
            this.neuronCounts = neuronCounts;
            return this;
        }

        public BPNeuralNetwork build() {
            logger.debug(String.format("Building network with %d inputs, %d hidden layers, %d outputs, bias=%f, defaultWeightValue is set to %b, transfer function is %s", inputsCount, neuronCounts.size(), outputsCount, bias, defaultWeightsValue, function.getClass().getCanonicalName()));
            BPNeuralNetwork result = new BPNeuralNetwork(inputsCount, outputsCount);
            setupLayers(result);
            return result;
        }

        private void setupLayers(BPNeuralNetwork result) {
            int inputs = inputsCount;
            result.layers = new ArrayList<>();
            for (Integer neuronCountInLayer : neuronCounts) {
                result.layers.add(buildLayer(inputs, neuronCountInLayer));
                inputs = neuronCountInLayer;
            }
            result.layers.add(buildLayer(inputs, outputsCount));
        }

        private Layer buildLayer(int inputs, Integer neuronCountInLayer) {
            Layer.LayerBuilder layerBuilder = new Layer.LayerBuilder(inputs, neuronCountInLayer).setTransferFunction(function);
            setupFreeParameters(layerBuilder);
            return layerBuilder.build();
        }

        private void setupFreeParameters(Layer.LayerBuilder layerBuilder) {
            setupWeights(layerBuilder);
            setupBias(layerBuilder);
        }

        private void setupBias(Layer.LayerBuilder layerBuilder) {
            if (randomizedBias)
                layerBuilder.randomizeBias();
            else
                layerBuilder.setBias(bias);
        }

        private void setupWeights(Layer.LayerBuilder layerBuilder) {
            if (defaultWeightsValue)
                layerBuilder.useDefaultWeights();
        }
    }

    public Layer getOutputLayer() {
        return layers.get(layers.size() - 1);
    }

    public Layer getHiddenLayer(int layerIndex) {
        checkArgument(layerIndex < layers.size() - 1);
        return layers.get(layerIndex);
    }

    public List<Double> giveResponse(List<Double> input) {
        checkArgument(input.size() == inputsCount, "Given input size must be the same as input count of the network");
        List<Double> response = input;
        for (Layer layer : layers) {
            response = layer.giveResponse(response);
        }
        logger.debug(String.format("Network response for %s is %s", input, response));
        return response;
    }
}

