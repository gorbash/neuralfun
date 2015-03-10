package com.gorbash.neural.network;

import com.gorbash.neural.tfunc.TransferFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Created by Gorbash on 2015-03-01.
 */
public class Layer {

    private static final double DEFAULT_WEIGHT = 1.0;

    private List<Neuron> neurons;

    int getNeuronsCount() {
        return neurons.size();
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    private Layer() {
    }

    static class LayerBuilder {
        private final int inputs;
        private final int outputs;
        private TransferFunction function;
        private double bias;
        private boolean defaultWeightsValue;
        private boolean randomizedBias;

        public LayerBuilder(int inputs, int outputs) {
            checkArgument(inputs > 0, "Count of inputs must be at least 1");
            checkArgument(outputs > 0, "Count of outputs must be at least 1");
            this.inputs = inputs;
            this.outputs = outputs;
        }

        public LayerBuilder setTransferFunction(TransferFunction function) {
            checkNotNull(function);
            this.function = function;
            return this;
        }

        public LayerBuilder setBias(double bias) {
            this.bias = bias;
            return this;
        }

        public LayerBuilder randomizeBias() {
            this.randomizedBias = true;
            return this;
        }

        public LayerBuilder useDefaultWeights() {
            defaultWeightsValue = true;
            return this;
        }

        public Layer build() {
            Layer result = new Layer();
            result.neurons = new ArrayList<>();
            setupNeurons(result);
            return result;
        }

        private void setupNeurons(Layer result) {
            for (int i = 0; i < outputs; i++) {
                Neuron.NeuronBuilder neuronBuilder = new Neuron.NeuronBuilder().setInputs(inputs).setTransferFunction(function);
                setupWeights(neuronBuilder);
                setupBias(neuronBuilder);
                result.neurons.add(neuronBuilder.build());
            }
        }

        private void setupBias(Neuron.NeuronBuilder neuronBuilder) {
            if (randomizedBias)
                neuronBuilder.randomizeBias();
            else
                neuronBuilder.setBias(bias);
        }

        private void setupWeights(Neuron.NeuronBuilder neuronBuilder) {
            if (defaultWeightsValue) {
                neuronBuilder.setWeights(getDefaultWeights(inputs));
            } else {
                neuronBuilder.randomizeWeights();
            }
        }

        private List<Double> getDefaultWeights(int inputs) {
            List<Double> weights = new ArrayList<>();
            for (int j = 0; j < inputs; j++) {
                weights.add(DEFAULT_WEIGHT);
            }
            return weights;
        }
    }

    List<Double> giveResponse(List<Double> stimuli) {
        return neurons.stream().map(n ->
                        applyStimuliAndGetResponse(n, stimuli)
        ).collect(Collectors.toList());
    }

    private Double applyStimuliAndGetResponse(Neuron n, List<Double> stimuli) {
        n.applyStimuli(stimuli);
        return n.getResponse();
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("Layer: ");
        for (Neuron neuron : neurons) {
            result.append("Neuron #").append(neuron.getNeuronID()).append(" ");
        }
        return result.toString();
    }
}
