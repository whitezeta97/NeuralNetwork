package com.contidavide;

import com.contidavide.activationfunctions.ActivationFunction;
import com.contidavide.costfunctions.CostFunction;

import java.util.*;
import java.util.stream.IntStream;
import java.util.Random;

public abstract class NeuralNetworkImpl implements NeuralNetwork {
    private final static String INPUTS_SIZE_NOT_EQUAL_TO_WEIGHTS_SIZE = "Inputs size not equal to weights size";

    protected ArrayList<Integer> numberNeuronsForLayer;
    protected int numberLayers;
    // [ [ Biases del livello 1 ], [ Biases del livello 2 ], ... ]
    protected ArrayList<ArrayList<Double>> biases;
    // [ [ livello 1 [ Weights del neurone 1 ], [ Weights del neurone 2 ], [ livello 2 [ Weights del neurone 1 ], [ Weights del neurone 2 ] ], ... ]
    protected ArrayList<ArrayList<ArrayList<Double>>> weights;
    protected ArrayList<ActivationFunction> activationsFunction;
    protected CostFunction costFunction;

    public NeuralNetworkImpl(final ArrayList<Integer> numberNeuronsForLayer,
                             final ArrayList<ActivationFunction> activationsFunction,
                             final CostFunction costFunction) {
        this.numberNeuronsForLayer = numberNeuronsForLayer;
        this.biases = this.getInitializedBiases();
        this.weights = this.getInitializedWeights();
        this.numberLayers = this.weights.size();
        this.activationsFunction = activationsFunction;
        this.costFunction = costFunction;
    }

    @Override
    public ArrayList<Double> feedforward(final ArrayList<Double> inputNeurons) {
        ArrayList<Double> outputNeurons = inputNeurons;
        for (int layer = 0; layer < this.numberLayers; layer++) {
            final ActivationFunction activationFunction = this.activationsFunction.get(layer);
            final ArrayList<ArrayList<Double>> neurons = this.getNeurons(layer);
            final int numberNeurons = neurons.size();
            final ArrayList<Double> neuronsActivations = new ArrayList<>();
            for (int neuron = 0; neuron < numberNeurons; neuron++) {
                final ArrayList<Double> neuronWeights = neurons.get(neuron);
                final Double bias = this.getBias(layer, neuron);
                final double z = this.computeScalarProduct(outputNeurons, neuronWeights) + bias;
                neuronsActivations.add(activationFunction.computeActivation(z));
            }
            outputNeurons = neuronsActivations;
        }

        return outputNeurons;
    }

    protected double computeScalarProduct(final ArrayList<Double> neuronInputs, final ArrayList<Double> neuronWeights) {
        if (neuronInputs.size() != neuronWeights.size()) {
            throw new RuntimeException(INPUTS_SIZE_NOT_EQUAL_TO_WEIGHTS_SIZE);
        }

        double z = 0;
        final int size = neuronInputs.size();
        for (int i = 0; i < size; i++) {
            z += neuronWeights.get(i) * neuronInputs.get(i);
        }

        return z;
    }

    protected int getNumberNeurons(final int layer) {
        return this.weights.get(layer).size();
    }

    protected Double getBias(final int layer, final int neuron) {
        return this.biases.get(layer).get(neuron);
    }

    protected ArrayList<ArrayList<Double>> getNeurons(final int layer) {
        return this.weights.get(layer);
    }

    protected ArrayList<Double> getNeuronWeights(final int layer, final int neuron) {
        return this.weights.get(layer).get(neuron);
    }

    protected Double getWeight(final int layer, final int neuron, final int weight) {
        return this.weights.get(layer).get(neuron).get(weight);
    }

    private ArrayList<ArrayList<Double>> getInitializedBiases() {
        // [ [ Biases del livello 1 ], [ Biases del livello 2 ], ... ]
        final ArrayList<ArrayList<Double>> biases = new ArrayList<>();

        final Random random = new Random();
        final int numberLayers = this.numberNeuronsForLayer.size();
        // layer = 1 perchè i neuroni di input non hanno il bias
        for (int layer = 1; layer < numberLayers; layer++) {
            final List<Double> neuronsBiases =
                    IntStream.range(0, this.numberNeuronsForLayer.get(layer)).mapToDouble(i -> random.nextGaussian()).boxed().toList();
            biases.add(new ArrayList<>(neuronsBiases));
        }

        return biases;
    }

    private ArrayList<ArrayList<ArrayList<Double>>> getInitializedWeights() {
        // [ [ livello 1 [ Weights del neurone 1 ], [ Weights del neurone 2 ], [ livello 2 [ Weights del neurone 1 ], [ Weights del neurone 2 ] ], ... ]
        ArrayList<ArrayList<ArrayList<Double>>> weights = new ArrayList<>();

        final Random random = new Random();
        final int numberLayers = this.numberNeuronsForLayer.size();
        // layer = 1 perchè i neuroni di input non hanno il bias
        for (int layer = 1; layer < numberLayers; layer++) {
            final ArrayList<ArrayList<Double>> neuronsWeigths = new ArrayList<>();
            final int numberWeights = this.numberNeuronsForLayer.get(layer - 1);

            for (int neuron = 0; neuron < this.numberNeuronsForLayer.get(layer); neuron++) {
                final List<Double> neuronWeights =
                        IntStream.range(0, numberWeights).mapToDouble(i -> random.nextGaussian()).boxed().toList();
                neuronsWeigths.add(new ArrayList<>(neuronWeights));
            }
            weights.add(neuronsWeigths);
        }

        return weights;
    }
}
