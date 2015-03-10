package com.gorbash.neural.trainer;

import org.apache.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static java.lang.Math.random;
import static java.lang.System.currentTimeMillis;

/**
 * Created by ars032 on 3/9/2015.
 */
public class TrainingManager {
    private final static Logger logger = Logger.getLogger(TrainingManager.class);
    private final Trainer trainer;
    private final int iterations;

    public TrainingManager(Trainer trainer) {
        this(trainer, 1000);
    }

    public TrainingManager(Trainer trainer, int iterations) {
        checkArgument(iterations > 0, "iterations must be greater than 0");
        this.iterations = iterations;
        this.trainer = checkNotNull(trainer);
    }

    public void runTraining(List<TrainingElement> trainingSuite) {
        long start = currentTimeMillis();
        prepareTestSuite(trainingSuite).stream().forEach(trainer::train);
        long end = currentTimeMillis();
        logger.info(String.format("Training of %s iterations, each %s steps took %s ms", iterations, trainingSuite.size(), (end-start)));
    }

    private List<TrainingElement> prepareTestSuite(List<TrainingElement> trainingSuite) {
        return getRandomizeSuite(getExplodedSuite(trainingSuite));
    }

    private List<TrainingElement> getRandomizeSuite(List<TrainingElement> initialTrainingList) {
        List<TrainingElement> randomizedTrainingList = new ArrayList<>();
        while(!initialTrainingList.isEmpty()) {
            randomizedTrainingList.add(popRandomElement(initialTrainingList));
        }
        return randomizedTrainingList;
    }

    private TrainingElement popRandomElement(List<TrainingElement> initialTrainingList) {
        return initialTrainingList.remove(getRandomElementIndex(initialTrainingList.size()));
    }

    private int getRandomElementIndex(int size) {
        return (int)(random()*size);
    }

    private List<TrainingElement> getExplodedSuite(List<TrainingElement> trainingSuite) {
        List<TrainingElement> initialTrainingList = new ArrayList<>();
        for (int i = 0; i < iterations; i++) {
            initialTrainingList.addAll(trainingSuite);
        }
        return initialTrainingList;
    }
}
