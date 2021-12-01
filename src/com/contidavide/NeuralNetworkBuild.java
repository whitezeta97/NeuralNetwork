package com.contidavide;

public interface NeuralNetworkBuild {
    NeuralNetworkBuild addLayer(final int numberNeurons) throws Exception;

    NeuralNetworkBuild addSigmoidActivationFunction() throws Exception;

    NeuralNetworkBuild addMeanSquareErrorCostFunction() throws Exception;

    NeuralNetworkImpl build() throws Exception;
}
