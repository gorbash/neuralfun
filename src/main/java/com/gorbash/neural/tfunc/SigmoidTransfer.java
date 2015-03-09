package com.gorbash.neural.tfunc;

/**
 * Created by Gorbash on 2015-02-03.
 * Sigmoid transfer function
 */
public class SigmoidTransfer implements TransferFunction {

    @Override
    public double transfer(double input) {
        return 1 / (1 + Math.exp(-input));
    }
}
