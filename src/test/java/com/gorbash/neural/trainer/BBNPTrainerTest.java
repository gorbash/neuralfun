package com.gorbash.neural.trainer;

import com.gorbash.neural.network.BPNeuralNetwork;
import com.gorbash.neural.trainer.BPNNTrainer;
import com.gorbash.neural.trainer.Trainer;
import com.gorbash.neural.trainer.TrainingElement;
import org.junit.Test;

import java.util.List;

import static java.lang.Math.abs;
import static java.util.Arrays.asList;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;

/**
 * Created by Gorbash on 2015-03-06.
 */
public class BBNPTrainerTest {

    @Test
    public void testThatOneLayerNetworkIsTrained() throws Exception {
        BPNeuralNetwork network = new BPNeuralNetwork.BPPNeuralNetworkBuilder(2, 1).randomizeBias().build();
        Trainer trainer = BPNNTrainer.build(network);
        List<Double> stimuli = asList(0.5, 0.5);
        double expectedOutput = 0.3;
        double before = network.giveResponse(stimuli).get(0);
        trainer.train(new TrainingElement(stimuli, asList(expectedOutput)));
        double after = network.giveResponse(stimuli).get(0);
        assertThat(abs(before - expectedOutput), greaterThan(abs(after - expectedOutput)));
    }

    @Test
    public void testThatTwoLayersNetworkIsTrained() throws Exception {
        BPNeuralNetwork network = new BPNeuralNetwork.BPPNeuralNetworkBuilder(2, 1).setHiddenLayers(asList(3, 2)).randomizeBias().build();
        Trainer trainer = BPNNTrainer.build(network);
        List<Double> stimuli = asList(0.5, 0.5);
        double expectedOutput = 0.3;
        double before = network.giveResponse(stimuli).get(0);
        trainer.train(new TrainingElement(stimuli, asList(expectedOutput)));
        double after = network.giveResponse(stimuli).get(0);
        assertThat(abs(before - expectedOutput), greaterThan(abs(after - expectedOutput)));
    }
}

