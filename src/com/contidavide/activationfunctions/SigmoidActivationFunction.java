package com.contidavide.activationfunctions;

public class SigmoidActivationFunction implements ActivationFunction {
    @Override
    public double computeActivation(final double z) {
        return 1.0 / (1.0 + Math.exp(-1 * z));
    }

    @Override
    public double computeDerivative(final double z) {
        final double activation = this.computeActivation(z);
        return activation * (1 - activation);
    }
}
