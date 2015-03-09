package com.gorbash.neural.tfunc;

/**
 * Created by Gorbash on 2015-02-03.
 * Linear Transfer function: f(x) = x
 */
public class LinearTransfer implements TransferFunction {
    @Override
    public double transfer(double input) {
        return input;
    }
}
