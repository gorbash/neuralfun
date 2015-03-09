package com.gorbash.neural.tfunc;

/**
 * Created by Gorbash on 2015-03-06.
 */
public class HardLimitTransfer implements TransferFunction {
    @Override
    public double transfer(double input) {
        return input < 0.5 ? 0 : 1;
    }
}
