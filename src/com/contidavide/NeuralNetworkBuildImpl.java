package com.contidavide;

import com.contidavide.activationfunctions.ActivationFunction;
import com.contidavide.costfunctions.CostFunction;
import com.contidavide.costfunctions.MeanSquareError;
import com.contidavide.optimizer.StochasticGradientDescent;

import java.util.ArrayList;

public class NeuralNetworkBuildImpl implements NeuralNetworkBuild {
    private static final String NEURAL_NETWORK_ALREADY_BUILD = "Non è possibile rifare il build, in quanto la rete è già stata costruita";
    private static final String COST_FUNCTION_ALREADY_BUILD = "La funzione di costo era già stata impostata";
    private static final String NEURAL_NETWORK_REQUIRES_AT_LEAST_THREE_LEVELS = "Non è possibile fare il build, in quanto la rete deve avere almeno 3 livelli";
    private static final String NEURAL_NETWORK_REQUIRES_COST_FUNCTION = "Non è possibile fare il build. in quanto non è stato impostata la funzione di costo";

    private boolean isBuild;
    private ArrayList<Integer> numberNeuronsForLayer;
    private ArrayList<ActivationFunction> activationsFunction;
    private CostFunction costFunction;

    public NeuralNetworkBuildImpl() {
        this.isBuild = false;
        this.numberNeuronsForLayer = new ArrayList<>();
        this.activationsFunction = new ArrayList<>();
    }

    @Override
    public NeuralNetworkBuild addLayer(final int numberNeurons, final ActivationFunction activationFunction) throws Exception {
        this.throwExceptionIfNeuralNetworkIsBuild();

        this.numberNeuronsForLayer.add(numberNeurons);
        this.activationsFunction.add(activationFunction);
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
    public StochasticGradientDescent buildStochasticGradientDescent() throws Exception {
        this.throwExceptionIfParametersNotSetted();

        return new StochasticGradientDescent(this.numberNeuronsForLayer, this.activationsFunction, this.costFunction);
    }

    private void throwExceptionIfParametersNotSetted() {
        this.throwExceptionIfNeuralNetworkIsBuild();
        if (this.numberNeuronsForLayer.size() < 3) {
            throw new RuntimeException(NEURAL_NETWORK_REQUIRES_AT_LEAST_THREE_LEVELS);
        }
        if (this.costFunction == null) {
            throw new RuntimeException(NEURAL_NETWORK_REQUIRES_COST_FUNCTION);
        }

        this.isBuild = true;
    }

    private void throwExceptionIfNeuralNetworkIsBuild() {
        if (isBuild) {
            throw new RuntimeException(NEURAL_NETWORK_ALREADY_BUILD);
        }
    }
}
