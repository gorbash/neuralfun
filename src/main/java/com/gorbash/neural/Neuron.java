package com.gorbash.neural;

import com.gorbash.neural.tfunc.LinearTransfer;
import com.gorbash.neural.tfunc.TransferFunction;
import org.apache.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Created by Gorbash on 2015-02-02.
 * Neuron
 */
public class Neuron {

    private final static Logger logger = Logger.getLogger(Neuron.class);

    private List<Double> weights;
    private List<Double> stimuli;
    private double response;
    private double bias;
    private TransferFunction tFunction;
    private long neuronID;


    public long getNeuronID() {
        return neuronID;
    }

    public int getInputsCount() {
        return weights.size();
    }

    public void applyStimuli(List<Double> stimuli) {
        this.stimuli = stimuli;
        recalculateResponse();
        logger.debug(String.format("Neuron #" + neuronID + ": applied stimuli %s, response is %f", stimuli, response));
    }

    private void recalculateResponse() {
        applyBias();
        calculateNetResponse();
        applyTransferFunction();
    }

    private void calculateNetResponse() {
        for (int i = 0; i < weights.size(); i++) {
            response += weights.get(i) * stimuli.get(i);
        }
    }

    private void applyBias() {
        response = bias;
    }

    private void applyTransferFunction() {
        response = tFunction.transfer(response);
    }

    public double getResponse() {
        return response;
    }

    public List<Double> getStimuli() {
        List<Double> ret = new ArrayList<>();
        ret.addAll(stimuli);
        return ret;
    }

    public List<Double> getWeights() {
        List<Double> ret = new ArrayList<>();
        ret.addAll(weights);
        return ret;
    }

    @Override
    public String toString() {
        return "Neuron #" + neuronID + ": {" +
                "weights=" + weights +
                ", stimuli=" + stimuli +
                ", response=" + response +
                ", bias=" + bias +
                ", tFunction=" + tFunction +
                '}';
    }

    private Neuron() {
    }

    public double getBias() {
        return bias;
    }

    public void setWeights(List<Double> weights) {
        checkArgument(this.weights.size() == weights.size());
        this.weights = weights;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public static class NeuronBuilder {

        private static int neuronIDGen = 1;
        private int inputsCount = 2;
        private List<Double> weights;
        private double bias;
        private TransferFunction transferFunction;
        private boolean randomizeWeights;

        public Neuron build() {
            Neuron ret = new Neuron();
            assignNeuronID(ret);
            setWeightsIfNotProvided(ret);
            setupBias(ret);
            setupTransferFunction(ret);
            applyInitialStimuli(ret);
            logger.debug(String.format("Build Neuron :%s", ret));
            return ret;
        }

        public NeuronBuilder randomizeWeights() {
            randomizeWeights = true;
            return this;
        }

        private void applyInitialStimuli(Neuron ret) {
            List<Double> stimuli = new ArrayList<>();
            for (int i = 0; i < inputsCount; i++) {
                stimuli.add(0.0);
            }
            ret.applyStimuli(stimuli);
        }

        private void setupBias(Neuron ret) {
            ret.bias = bias;
        }

        private void setupTransferFunction(Neuron ret) {
            if (transferFunction == null)
                ret.tFunction = new LinearTransfer();
            else
                ret.tFunction = transferFunction;
        }

        private void assignNeuronID(Neuron ret) {
            ret.neuronID = neuronIDGen++;
        }

        private void setWeightsIfNotProvided(Neuron ret) {
            if (weights == null) {
                weights = new ArrayList<>();
                for (int i = 0; i < inputsCount; i++) {
                    if (randomizeWeights)
                        weights.add(getRandomNumber());
                    else
                        weights.add(0.0);
                }
            }
            ret.weights = weights;
        }

        private double getRandomNumber() {
            return (Math.random() * 2) - 1;
        }

        public NeuronBuilder setInputs(int inputs) {
            checkArgument(inputs > 0, "Count of inputs must be at least 1.");
            inputsCount = inputs;
            return this;
        }

        public NeuronBuilder setWeights(List<Double> weights) {
            this.weights = checkNotNull(weights);
            return this;
        }

        public NeuronBuilder setBias(double bias) {
            this.bias = bias;
            return this;
        }

        public NeuronBuilder randomizeBias() {
            this.bias = getRandomNumber();
            return this;
        }

        public NeuronBuilder setTransferFunction(TransferFunction transferFunction) {
            this.transferFunction = checkNotNull(transferFunction);
            return this;
        }
    }
}
