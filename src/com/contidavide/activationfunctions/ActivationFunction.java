package com.contidavide.activationfunctions;

public interface ActivationFunction {
    double computeActivation(final double z);

    double computeDerivative(final double z);
}
