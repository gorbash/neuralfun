package com.gorbash.neural.network;

import java.util.List;

/**
 * Created by Gorbash on 2015-03-08.
 */
public interface NeuralNetwork {
    public List<Double> giveResponse(List<Double> input);
}
