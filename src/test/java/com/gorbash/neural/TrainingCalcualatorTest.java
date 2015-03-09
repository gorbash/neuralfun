package com.gorbash.neural;

import org.junit.Test;

import java.util.Arrays;

import static java.util.Arrays.asList;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;

/**
 * Created by Gorbash on 2015-03-08.
 */
public class TrainingCalcualatorTest {

    TrainingCalculator calc = new TrainingCalculator(2.0);

    @Test
    public void testOutputLayerErrorFactors() throws Exception {
        assertThat(calc.calculateOutputLayerErrorFactors(asList(2.0, 3.0), asList(-1.0, -2.0)), is(asList(-3.0, -5.0)));
    }

    @Test
    public void testCalculateDeltas() throws Exception {
        assertThat(calc.calculateDeltas(asList(0.5, 0.3), asList(5.0, 7.0)), is(asList(1.25, 1.47)));
    }

    @Test
    public void testCalculateNewBias() throws Exception {
        assertThat(calc.calculateNewBias(3.0, 5.0), is(13.0));

    }

    @Test
    public void testCalculateNewWeights() throws Exception {
        assertThat(calc.calculateNewWeights(asList(1.0, -2.0), asList(3.0, 7.0), 5.0), is(asList(13.0, -13.0)));
    }

}
