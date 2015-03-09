package com.gorbash.neural;

import org.junit.Test;

import static java.util.Arrays.asList;
import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

/**
 * Created by Gorbash on 2015-03-08.
 */
public class NeuralNetworkMonitorTest {

    private final NeuralNetwork network = input -> asList(1.0, 2.0, 3.0);

    private final NeuralNetworkMonitor monitor = new NeuralNetworkMonitor(network);

    @Test
    public void testThatMonitorCanCalculateTotalErrorForOneElement() throws Exception {
        assertThat(monitor.getTotalError(asList(
                        new TrainingElement(asList(0.0, 0.0, 0.0), asList(0.0, 0.0, 0.0)))),
                is(6.0));
    }

    @Test
    public void testThatMonitorCanCalculateTotalErrorForThreeElement() throws Exception {
        assertThat(monitor.getTotalError(asList(
                        new TrainingElement(asList(0.0, 0.0, 0.0), asList(0.5, 0.5, 0.5)),      //4.5
                        new TrainingElement(asList(1.0, 2.0, 3.0), asList(-1.0, -2.0, -2.0)),   //11
                        new TrainingElement(asList(-2.0, 6.0, -4.0), asList(1.0, 1.0, 3.0)))),  //1
                is(16.5));
    }

    @Test
    public void testThatMonitorCanCalculateTotalErrorForLearntNetwork() throws Exception {
        assertThat(monitor.getTotalError(asList(
                        new TrainingElement(asList(0.0, 0.0, 0.0), asList(1.0, 2.0, 3.0)))),
                is(0.0));
    }

}
