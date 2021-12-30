package com.contidavide.optimizer;

import com.contidavide.activationfunctions.ActivationFunction;
import com.contidavide.costfunctions.CostFunction;

import java.util.ArrayList;

public class StochasticGradientDescent extends OptimezerImpl {

    public StochasticGradientDescent(final ArrayList<Integer> numberNeuronsForLayer,
                                     final ArrayList<ActivationFunction> activationsFunction,
                                     final CostFunction costFunction) {
        super(numberNeuronsForLayer, activationsFunction, costFunction);
    }
}
