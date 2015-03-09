package com.gorbash.neural;

import java.util.ArrayList;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Created by ars032 on 3/9/2015.
 */
public class TrainingManager {

    private Trainer trainer;
    private final int iterations;

    public TrainingManager(Trainer trainer) {
        this(trainer, 10);
    }

    public TrainingManager(Trainer trainer, int iterations) {
        checkArgument(iterations > 0, "iterations must be greater than 0");
        this.iterations = iterations;
        this.trainer = checkNotNull(trainer);
    }

    public void runTraining(List<TrainingElement> trainingSuite) {
        List<TrainingElement> initialTrainingList = new ArrayList<>();

        for (int i = 0; i < iterations; i++) {
            initialTrainingList.addAll(trainingSuite);
        }
        List<TrainingElement> randomizedTrainingList = new ArrayList<>();

        while(!initialTrainingList.isEmpty()) {
            TrainingElement randomElement = initialTrainingList.remove((int)(Math.random()*initialTrainingList.size()));
            randomizedTrainingList.add(randomElement);
        }

        randomizedTrainingList.stream().forEach(t -> trainer.train(t));
    }
}
