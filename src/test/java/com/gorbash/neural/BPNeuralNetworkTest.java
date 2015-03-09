package com.gorbash.neural;

import com.gorbash.neural.tfunc.LinearTransfer;
import org.junit.Test;

import java.util.List;

import static java.util.Arrays.asList;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;
import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertEquals;

/**
 * Created by Gorbash on 2015-02-18.
 */
public class BPNeuralNetworkTest {

    @Test
    public void testThatNetworkCanGiveOutput() {
        BPNeuralNetwork network = new BPNeuralNetwork.BPPNeuralNetworkBuilder(3, 4).setBias(0.5).setTransferFunction(new LinearTransfer()).defaultWeightsValue().build();
        assertThat(network.getHiddenLayersCount(), is(0));
        List<Double> output = network.giveResponse(asList(1.0, 2.0, 3.0));
        assertThat(output, hasSize(4));
        assertEquals(asList(6.5, 6.5, 6.5, 6.5), output);
    }

    @Test
    public void testThatHiddenLayerCanParticipateInGivingOutput() throws Exception {
        BPNeuralNetwork network = new BPNeuralNetwork.BPPNeuralNetworkBuilder(2, 1).setBias(0.5).setTransferFunction(new LinearTransfer()).setHiddenLayers(asList(2, 3)).defaultWeightsValue().build();
        assertThat(network.getHiddenLayersCount(), is(2));
        assertThat(network.getInputsCount(), is(2));
        assertThat(network.getOutputsCount(), is(1));
        assertThat(network.getNeuronsCountInLayer(0), is(2));
        assertThat(network.getNeuronsCountInLayer(1), is(3));
        assertThat(network.getNeuronsCountInLayer(2), is(1));

        assertEquals(asList(29.0), network.giveResponse(asList(2.0, 2.0)));

        assertEquals(asList(-19.0), network.giveResponse(asList(-2.0, -2.0)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testThatHiddenLayerCannotHaveNegativeNumberOfNeurons() throws Exception {
        new BPNeuralNetwork.BPPNeuralNetworkBuilder(3, 4).setBias(0.5).setTransferFunction(new LinearTransfer()).setHiddenLayers(asList(2, -1)).defaultWeightsValue().build();
    }

    @Test
    public void testThatSigmoidBPNetworkGivesOutputInRange() throws Exception {
        BPNeuralNetwork neuralNetwork = new BPNeuralNetwork.BPPNeuralNetworkBuilder(3, 3).setBias(Math.random()).setHiddenLayers(asList(17, 34, 23, 12, 56)).build();
        List<Double> response = neuralNetwork.giveResponse(asList(Math.random() * 1000 - 500, Math.random() * 1000 - 500, Math.random() * 1000 - 500));
        assertThat(response, hasSize(3));
        for (Double el : response)
            assertThat(el, allOf(greaterThan(0.0), lessThan(1.0)));

    }

    @Test
    public void testGetHiddenLayerReturnsHiddenLayers() throws Exception {
        BPNeuralNetwork neuralNetwork = new BPNeuralNetwork.BPPNeuralNetworkBuilder(3, 3).setBias(Math.random()).setHiddenLayers(asList(17, 34, 23, 12, 56)).build();
        assertThat(neuralNetwork.getHiddenLayer(0).getNeuronsCount(), is(17));
        assertThat(neuralNetwork.getHiddenLayer(1).getNeuronsCount(), is(34));
        assertThat(neuralNetwork.getHiddenLayer(2).getNeuronsCount(), is(23));
        assertThat(neuralNetwork.getHiddenLayer(3).getNeuronsCount(), is(12));
        assertThat(neuralNetwork.getHiddenLayer(4).getNeuronsCount(), is(56));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testThatGetHiddenLayersWillNotReturnOutputLayer() throws Exception {
        BPNeuralNetwork neuralNetwork = new BPNeuralNetwork.BPPNeuralNetworkBuilder(3, 3).setBias(Math.random()).setHiddenLayers(asList(17, 34, 23, 12, 56)).build();
        neuralNetwork.getHiddenLayer(5);
    }
}
