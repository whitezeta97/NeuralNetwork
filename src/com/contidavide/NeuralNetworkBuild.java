package com.contidavide;

import com.contidavide.activationfunctions.ActivationFunction;
import com.contidavide.optimizer.OptimezerImpl;
import com.contidavide.optimizer.StochasticGradientDescent;

public interface NeuralNetworkBuild {
    NeuralNetworkBuild addLayer(final int numberNeurons, final ActivationFunction activationFunction) throws Exception;

    NeuralNetworkBuild addMeanSquareErrorCostFunction() throws Exception;

    StochasticGradientDescent buildStochasticGradientDescent() throws Exception;
}
