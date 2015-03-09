package com.gorbash.neural;

import org.junit.Test;

import java.util.*;

import static java.util.Arrays.asList;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;
import static org.hamcrest.core.IsNot.not;

/**
 * Created by ars032 on 3/9/2015.
 */
public class TrainingManagerTest {

    @Test
    public void testThatManagerUsesAllElements() throws Exception {

        List<TrainingElement> trainList = asList(
                new TrainingElement(asList(1.0, 2.0), asList(2.0, 3.0)),
                new TrainingElement(asList(0.0, 1.0), asList(-3.0, 12.0)),
                new TrainingElement(asList(-4.0, 66.0), asList(0.0, 0.0))
        );

        final Set<TrainingElement> trained = new HashSet<>();
        Trainer trainer = trained::add;
        TrainingManager trainingManager = new TrainingManager(trainer);
        trainingManager.runTraining(trainList);
        assertThat(trained, is(new HashSet<>(trainList)));
    }

    @Test
    public void testThatManagerTestsManyTimes() throws Exception {
        List<TrainingElement> trainList = asList(
                new TrainingElement(asList(1.0, 2.0), asList(2.0, 3.0)),
                new TrainingElement(asList(0.0, 1.0), asList(-3.0, 12.0))
        );
        List<TrainingElement> trained = new ArrayList<>();
        Trainer trainer = trained::add;

        TrainingManager trainingManager = new TrainingManager(trainer, 3);
        trainingManager.runTraining(trainList);

        assertThat(trained.size(), is(6));
        assertThat(new HashSet<>(trained), is(new HashSet<>(trainList)));
    }

    @Test
    public void testThatManagerRandomizeTestSuite() throws Exception {
        List<TrainingElement> trainList = asList(
                new TrainingElement(asList(1.0, 2.0), asList(2.0, 3.0)),
                new TrainingElement(asList(0.0, 1.0), asList(-3.0, 12.0))
        );
        final int  iterationsCount = 1000;

        List<TrainingElement> trained = new ArrayList<>();
        Trainer trainer = trained::add;


        TrainingManager trainingManager = new TrainingManager(trainer, iterationsCount);
        trainingManager.runTraining(trainList);

        List<TrainingElement> multipliedList = new ArrayList<>();
        for (int i = 0; i < iterationsCount; i++) {
            multipliedList.addAll(trainList);
        }

        assertThat(trained.size(), is(iterationsCount*trainList.size()));
        assertThat(trained, not(multipliedList));
    }
}
