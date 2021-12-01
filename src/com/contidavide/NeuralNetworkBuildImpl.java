package com.contidavide;

import com.contidavide.activationfunctions.ActivationFunction;
import com.contidavide.activationfunctions.SigmoidActivationFunction;
import com.contidavide.costfunctions.CostFunction;
import com.contidavide.costfunctions.MeanSquareError;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class NeuralNetworkBuildImpl implements NeuralNetworkBuild {
    private static final String NEURAL_NETWORK_ALREADY_BUILD = "Non è possibile rifare il build, in quanto la rete è già stata costruita";
    private static final String ACTIVATION_FUNCTION_ALREADY_BUILD = "La funzione di attivazione era già stata impostata";
    private static final String COST_FUNCTION_ALREADY_BUILD = "La funzione di costo era già stata impostata";
    private static final String NEURAL_NETWORK_REQUIRES_AT_LEAST_THREE_LEVELS = "Non è possibile fare il build, in quanto la rete deve avere almeno 3 livelli";
    private static final String NEURAL_NETWORK_REQUIRES_ACTIVATION_FUNCTION = "Non è possibile fare il build. in quanto non è stato impostata la funzione di attivazione";
    private static final String NEURAL_NETWORK_REQUIRES_COST_FUNCTION = "Non è possibile fare il build. in quanto non è stato impostata la funzione di costo";

    private boolean isBuild;
    private ArrayList<Integer> numberNeuronsForLayer;
    private ActivationFunction activationFunction;
    private CostFunction costFunction;

    public NeuralNetworkBuildImpl() {
        this.isBuild = false;
        this.numberNeuronsForLayer = new ArrayList<>();
    }

    @Override
    public NeuralNetworkBuild addLayer(final int numberNeurons) throws Exception {
        this.throwExceptionIfNeuralNetworkIsBuild();

        this.numberNeuronsForLayer.add(numberNeurons);
        return this;
    }

    @Override
    public NeuralNetworkBuild addSigmoidActivationFunction() throws Exception {
        this.throwExceptionIfNeuralNetworkIsBuild();
        if (this.activationFunction != null) {
            throw new RuntimeException(ACTIVATION_FUNCTION_ALREADY_BUILD);
        }

        this.activationFunction = new SigmoidActivationFunction();
        return this;
    }

    @Override
    public NeuralNetworkBuild addMeanSquareErrorCostFunction() throws Exception {
        this.throwExceptionIfNeuralNetworkIsBuild();
        if (this.costFunction != null) {
            throw new RuntimeException(COST_FUNCTION_ALREADY_BUILD);
        }

        this.costFunction = new MeanSquareError();
        return this;
    }

    @Override
    public NeuralNetworkImpl build() throws Exception {
        this.throwExceptionIfNeuralNetworkIsBuild();
        if (this.numberNeuronsForLayer.size() < 3) {
            throw new RuntimeException(NEURAL_NETWORK_REQUIRES_AT_LEAST_THREE_LEVELS);
        }
        if (this.activationFunction == null) {
            throw new RuntimeException(NEURAL_NETWORK_REQUIRES_ACTIVATION_FUNCTION);
        }
        if (this.costFunction == null) {
            throw new RuntimeException(NEURAL_NETWORK_REQUIRES_COST_FUNCTION);
        }

        this.isBuild = true;
        return new NeuralNetworkImpl(this.numberNeuronsForLayer, this.activationFunction, this.costFunction);
    }

    private void throwExceptionIfNeuralNetworkIsBuild() {
        if (isBuild) {
            throw new RuntimeException(NEURAL_NETWORK_ALREADY_BUILD);
        }
    }
}
