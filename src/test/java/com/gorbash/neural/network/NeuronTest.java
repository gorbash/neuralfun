package com.gorbash.neural.network;

import com.cedarsoftware.util.io.JsonReader;
import com.cedarsoftware.util.io.JsonWriter;
import com.gorbash.neural.network.Neuron;
import com.gorbash.neural.tfunc.SigmoidTransfer;
import com.gorbash.neural.tfunc.TransferFunction;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;

import static java.util.Arrays.asList;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.core.Is.is;

/**
 * Created by Gorbash on 2015-02-02.
 */
public class NeuronTest {

    private final double epsilon = 0.0000000001;

    @Test
    public void testThatNeuronGivesDefaultOutput() throws Exception {
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(3).build();
        Assert.assertEquals(3, neuron.getInputsCount());
        Assert.assertEquals(0.0, neuron.getResponse(), epsilon);
    }

    @Test
    public void testThatInputsCanBeObtained() throws Exception {
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(2).build();
        neuron.applyStimuli(asList(13.0, 15.0));
        assertThat(neuron.getStimuli(), is(asList(13.0, 15.0)));
    }

    @Test
    public void testThatWeightsCanBeObtained() throws Exception {
        List<Double> weights = asList(21.0, -3.0);
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(2).setWeights(weights).build();
        assertThat(neuron.getWeights(), is(weights));
    }

    @Test
    public void testThatNeuronRespondsForInputs() throws Exception {
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(2).setWeights(asList(3.0, 5.0)).build();
        neuron.applyStimuli(asList(7.0, 11.0));
        Assert.assertEquals(76.0, neuron.getResponse(), epsilon);
    }

    @Test
    public void testThatNeronCanCalculateResponseTwice() throws Exception {
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(2).setWeights(asList(3.0, 5.0)).build();
        neuron.applyStimuli(asList(1.0, 2.0));
        Assert.assertEquals(13.0, neuron.getResponse(), epsilon);
        neuron.applyStimuli(asList(5.0, 4.0));
        Assert.assertEquals(35.0, neuron.getResponse(), epsilon);
    }

    @Test
    public void testNeuronCanBeSavedToJSON() throws Exception {
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(2).setWeights(asList(3.0, 5.0)).build();
        neuron.applyStimuli(asList(1.0, 2.0));
        String json = JsonWriter.objectToJson(neuron);
        Neuron otherNeuron = (Neuron) JsonReader.jsonToJava(json);
        assertThat(otherNeuron.getResponse(), is(neuron.getResponse()));
        assertThat(otherNeuron.getInputsCount(), is(neuron.getInputsCount()));
        assertThat(otherNeuron.getStimuli(), is(neuron.getStimuli()));
        assertThat(otherNeuron.getWeights(), is(neuron.getWeights()));


        List<Double> otherStimuli = asList(0.5, 13.0);
        neuron.applyStimuli(otherStimuli);
        otherNeuron.applyStimuli(otherStimuli);

        assertThat(otherNeuron.getResponse(), is(neuron.getResponse()));


    }

    @Test
    public void testThatNeuronCanHaveBias() throws Exception {
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(2).setBias(0.5).build();
        neuron.applyStimuli(asList(0.0, 0.0));
        assertThat(neuron.getResponse(), is(0.5));
    }

    @Test
    public void testThatNeuronCanHaveBiaAndWeights() throws Exception {
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(2).setBias(0.5).setWeights(asList(-2.0, 13.0)).build();
        neuron.applyStimuli(asList(3.0, 2.0));
        assertThat(neuron.getResponse(), is(20.5));
    }


    @Test
    public void testThatNeuronCanHaveSigmoidTransferFunction() throws Exception {
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(2).setWeights(asList(1.0, 1.0)).setTransferFunction(new SigmoidTransfer()).build();
        neuron.applyStimuli(asList(0.0, 0.0));
        assertThat(neuron.getResponse(), is(0.5));

        neuron.applyStimuli(asList(100000.0, 1000000.0));
        assertThat(neuron.getResponse(), is(closeTo(1.0, epsilon)));

        neuron.applyStimuli(asList(-100000.0, -1000000.0));
        assertThat(neuron.getResponse(), is(closeTo(0, epsilon)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testIncorrectInputs() throws Exception {
        new Neuron.NeuronBuilder().setInputs(0);
    }

    @Test(expected = NullPointerException.class)
    public void testNullTransferFunction() throws Exception {
        new Neuron.NeuronBuilder().setTransferFunction(null);
    }

    @Test(expected = NullPointerException.class)
    public void testNullWeights() throws Exception {
        new Neuron.NeuronBuilder().setWeights(null);
    }

    @Test
    public void testThatWeightsCanBeRandomized() throws Exception {
        Neuron neuron = new Neuron.NeuronBuilder().setInputs(3).randomizeWeights().build();
        assertThat(neuron.getWeights().get(0), not(is(neuron.getWeights().get(1))));
        assertThat(neuron.getWeights().get(0), not(is(neuron.getWeights().get(2))));
        assertThat(neuron.getWeights().get(1), not(is(neuron.getWeights().get(2))));
    }
}
