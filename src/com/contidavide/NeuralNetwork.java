package com.contidavide;

import java.util.ArrayList;
import java.util.Optional;

public interface NeuralNetwork {
    ArrayList<Double> feedforward(ArrayList<Double> inputNeurons);

    void trainWithStochasticGradientDescent(final ArrayList<Data> trainData, final Optional<ArrayList<Data>> testData,
                                            final int numberEpochs, final int miniBatchLength,
                                            final double learningRate);
}
