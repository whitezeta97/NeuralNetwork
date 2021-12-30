package com.contidavide.optimizer;

import com.contidavide.Sample;

import java.util.ArrayList;
import java.util.Optional;

public interface Optimizer {
    void train(final ArrayList<Sample> trainData, final Optional<ArrayList<Sample>> testData,
               final int numberEpochs, final int miniBatchLength, final double learningRate);
}
