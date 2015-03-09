package com.gorbash.neural;

import java.util.List;

/**
 * Created by Gorbash on 2015-03-08.
 */
public class TrainingElement {

    private final List<Double> input;
    private final List<Double> expectedOutput;

    public TrainingElement(List<Double> input, List<Double> expectedOutput) {
        this.input = input;
        this.expectedOutput = expectedOutput;
    }

    public List<Double> getExpectedOutput() {
        return expectedOutput;
    }

    public List<Double> getInput() {
        return input;
    }

}
