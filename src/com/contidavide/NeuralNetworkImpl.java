package com.contidavide;

import com.contidavide.activationfunctions.ActivationFunction;
import com.contidavide.costfunctions.CostFunction;

import java.util.*;
import java.util.stream.IntStream;
import java.util.Random;

public class NeuralNetworkImpl implements NeuralNetwork {
    private final static String INPUTS_SIZE_NOT_EQUAL_TO_WEIGHTS_SIZE = "Inputs size not equal to weights size";

    private ArrayList<Integer> numberNeuronsForLayer;
    private int numberLayers;
    // [ [ Biases del livello 1 ], [ Biases del livello 2 ], ... ]
    private ArrayList<ArrayList<Double>> biases;
    // [ [ livello 1 [ Weights del neurone 1 ], [ Weights del neurone 2 ], [ livello 2 [ Weights del neurone 1 ], [ Weights del neurone 2 ] ], ... ]
    private ArrayList<ArrayList<ArrayList<Double>>> weights;
    private ActivationFunction activationFunction;
    private CostFunction costFunction;

    public NeuralNetworkImpl(final ArrayList<Integer> numberNeuronsForLayer,
                             final ActivationFunction activationFunction,
                             final CostFunction costFunction) {
        this.numberNeuronsForLayer = numberNeuronsForLayer;
        this.biases = this.getInitializedBiases();
        this.weights = this.getInitializedWeights();
        this.numberLayers = this.weights.size();
        this.activationFunction = activationFunction;
        this.costFunction = costFunction;
    }

    @Override
    public ArrayList<Double> feedforward(final ArrayList<Double> inputNeurons) {
        ArrayList<Double> outputNeurons = inputNeurons;
        for (int layer = 0; layer < this.numberLayers; layer++) {
            final ArrayList<ArrayList<Double>> neurons = this.getNeurons(layer);
            final int numberNeurons = neurons.size();
            final ArrayList<Double> neuronsActivations = new ArrayList<>();
            for (int neuron = 0; neuron < numberNeurons; neuron++) {
                final ArrayList<Double> neuronWeights = neurons.get(neuron);
                final Double bias = this.getBias(layer, neuron);
                final double z = this.computeScalarProduct(outputNeurons, neuronWeights) + bias;
                neuronsActivations.add(this.activationFunction.computeActivation(z));
            }
            outputNeurons = neuronsActivations;
        }

        return outputNeurons;
    }

    @Override
    public void trainWithStochasticGradientDescent(final ArrayList<Data> trainData, final Optional<ArrayList<Data>> testData,
                                                   final int numberEpochs, final int miniBatchLength,
                                                   final double learningRate) {
        for (int epoch = 0; epoch < numberEpochs; epoch++) {
            Collections.shuffle(trainData);
            final ArrayList<ArrayList<Data>> miniBatches = getMiniBatches(trainData, miniBatchLength);
            for (ArrayList<Data> miniBatch : miniBatches) {
                this.computeGradientsAndUpdateBiasesAndWeights(miniBatch, learningRate);
            }

            if (testData.isPresent()) {
                System.out.println(String.format("Epoch %s complete: %s / %s", epoch, this.evaluate(testData.get()), testData.get().size()));
            } else {
                System.out.println(String.format("Epoch %s complete", epoch));
            }
        }
    }

    private double computeScalarProduct(final ArrayList<Double> neuronInputs, final ArrayList<Double> neuronWeights) {
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

    private ArrayList<ArrayList<Data>> getMiniBatches(final ArrayList<Data> trainData, final int miniBatchLength) {
        final ArrayList<ArrayList<Data>> miniBatches = new ArrayList<>();
        ArrayList<Data> miniBatch = new ArrayList<>();
        int numberElementsInMiniBatch = 0;

        for (Data data : trainData) {
            miniBatch.add(data);
            numberElementsInMiniBatch++;
            if (numberElementsInMiniBatch == miniBatchLength) {
                miniBatches.add(miniBatch);
                miniBatch = new ArrayList<>();
                numberElementsInMiniBatch = 0;
            }
        }
        if (!miniBatch.isEmpty()) {
            miniBatches.add(miniBatch);
        }

        return miniBatches;
    }

    /**
     * Calcola la somma di tutti i gradienti dei biases e dei weights sui dati del minibatch ed infine aggiorna i pesi dei biases e dei weights
     * @param miniBatch
     * @param learningRate
     */
    private void computeGradientsAndUpdateBiasesAndWeights(final ArrayList<Data> miniBatch, final double learningRate) {
        // Calcolo la somma di tutti i gradienti dei biases e dei weights sui dati del minibatch
        final Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>> biasesAndWeightsGradients = this.computeBiasesAndWeightsGradients(miniBatch);
        final ArrayList<ArrayList<Double>> biasesGradients = biasesAndWeightsGradients.getElement1();
        final ArrayList<ArrayList<ArrayList<Double>>> weightsGradients = biasesAndWeightsGradients.getElement2();
        // Aggiorno i pesi dei biases e dei weights
        this.updateBiasesAndWeights(biasesGradients, weightsGradients, miniBatch.size(), learningRate);
    }

    /**
     * Aggiorna i pesi dei biases e dei weights
     * @param biasesGradients somma di tutti i gradienti dei biases effettuati sul minibatch
     * @param weightsGradients somma di tutti i gradienti dei weights effettuati sul minibatch
     * @param miniBatchSize grandezza del miniBatch
     * @param learningRate
     */
    private void updateBiasesAndWeights(final ArrayList<ArrayList<Double>> biasesGradients,
                                        final ArrayList<ArrayList<ArrayList<Double>>> weightsGradients,
                                        final int miniBatchSize, final double learningRate) {
        for (int layer = 0; layer < this.numberLayers; layer++) {
            final int numberNeurons = this.getNumberNeurons(layer);
            for (int neuron = 0; neuron < numberNeurons; neuron++) {
                // Update Bias
                final double biasGradient = biasesGradients.get(layer).get(neuron);
                final Double currentBias = this.getBias(layer, neuron);
                final Double updatedBias = currentBias - (learningRate / miniBatchSize) * biasGradient;
                this.biases.get(layer).set(neuron, updatedBias);

                final int numberWeights = this.getNeuronWeights(layer, neuron).size();
                for (int weight = 0; weight < numberWeights; weight++) {
                    // Update weight
                    final Double weightGradient = weightsGradients.get(layer).get(neuron).get(weight);
                    final Double currentWeight = this.getWeight(layer, neuron, weight);
                    final Double updatedWeight = currentWeight - (learningRate / miniBatchSize) * weightGradient;
                    this.weights.get(layer).get(neuron).set(weight, updatedWeight);
                }
            }
        }
    }

    /**
     * Calcola la somma di tutti i gradienti dei biases e dei weights sui dati del minibatch.
     * Per ogni dato nel minibatch si effettua la backpropagation per calcolare i gradienti. Successivamente i gradienti ottenuti vengono sommati ai gradienti totali
     * @param miniBatch
     * @return la somma di tutti i gradienti dei biases e dei weights effettuata sul minibatch
     * Il tipo di ritorno Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>>: elemento1 contiene i valori dei gradienti dei biases ed elemento2 contiene i valori dei gradienti dei weights
     */
    private Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>> computeBiasesAndWeightsGradients(final ArrayList<Data> miniBatch) {
        final ArrayList<ArrayList<Double>> totalBiasesGradients = this.getZeroBiases();
        final ArrayList<ArrayList<ArrayList<Double>>> totalWeightsGradients = this.getZeroWeights();

        for (Data data : miniBatch) {
            Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>> biasesAndWeightsGradients = this.backpropagation(data);
            final ArrayList<ArrayList<Double>> biasesGradientsOfBackpropagation = biasesAndWeightsGradients.getElement1();
            final ArrayList<ArrayList<ArrayList<Double>>> weightsGradientsOfBackpropagation = biasesAndWeightsGradients.getElement2();

            // Update gradients of Biases and Weights
            for (int layer = 0; layer < this.numberLayers; layer++) {
                final int numberNeurons = this.getNumberNeurons(layer);
                for (int neuron = 0; neuron < numberNeurons; neuron++) {
                    // Update gradients of Biases
                    final Double totalBiasGradient = totalBiasesGradients.get(layer).get(neuron);
                    final Double biasGradientOfBackPropagation = biasesGradientsOfBackpropagation.get(layer).get(neuron);
                    totalBiasesGradients.get(layer).set(neuron, totalBiasGradient + biasGradientOfBackPropagation);

                    // Update gradients of Weights
                    final int numberWeights = this.getNeuronWeights(layer, neuron).size();
                    final ArrayList<Double> totalWeightsGradientsOfNeuron = totalWeightsGradients.get(layer).get(neuron);
                    final ArrayList<Double> weightsGradientsOfBackPropagation = weightsGradientsOfBackpropagation.get(layer).get(neuron);
                    for (int weight = 0; weight < numberWeights; weight++) {
                        final Double totalWeightGradient = totalWeightsGradientsOfNeuron.get(weight);
                        final Double weightGradientOfBackPropagation = weightsGradientsOfBackPropagation.get(weight);
                        totalWeightsGradientsOfNeuron.set(weight, totalWeightGradient + weightGradientOfBackPropagation);
                    }
                }
            }
        }

        return new Pair<>(totalBiasesGradients, totalWeightsGradients);
    }

    private ArrayList<ArrayList<Double>> getZeroBiases() {
        final ArrayList<ArrayList<Double>> zeroBiases = new ArrayList<>();
        this.biases.forEach(layerBiases -> zeroBiases.add(new ArrayList<>(Collections.nCopies(layerBiases.size(), 0.0))));

        return zeroBiases;
    }

    private ArrayList<ArrayList<ArrayList<Double>>> getZeroWeights() {
        final ArrayList<ArrayList<ArrayList<Double>>> zeroWeights = new ArrayList<>();
        this.weights.forEach(layerWeights -> {
            final ArrayList<ArrayList<Double>> weights = new ArrayList<>();
            layerWeights.forEach(neuronsWeights -> weights.add(new ArrayList<>(Collections.nCopies(neuronsWeights.size(), 0.0))));
            zeroWeights.add(weights);
        });

        return zeroWeights;
    }

    /**
     * Algoritmo di backpropagation
     *
     * @return tutti i gradienti dei biases e dei weights.
     * Il tipo di ritorno Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>>: elemento1 contiene i valori dei gradienti dei biases ed elemento2 contiene i valori dei gradienti dei weights
     * */
    private Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>> backpropagation(final Data data) {
        // Calcolo di tutti i valori z e tutti i valori di attivazione di ogni neurone, livello per livello
        final Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<Double>>> zsAndActivations =
                this.feedforwardAndGetAllZsAndActivations(data.geDataToDouble());
        final ArrayList<ArrayList<Double>> zs = zsAndActivations.getElement1();
        final ArrayList<ArrayList<Double>> activations = zsAndActivations.getElement2();

        // Prima equazione fondamentale: Calcolo tutti i valori Delta, di ogni neurone, dell'ultimo layer
        final ArrayList<Double> zsOfLastLayer = zs.get(zs.size() - 1);
        final ArrayList<Double> activationsOfLastLayer = activations.get(activations.size() - 1);
        ArrayList<Double> deltas = this.getDeltaOfLastLayer(data.getLabelToDouble(), zsOfLastLayer, activationsOfLastLayer);

        // Calcoliamo tutti i valori dei gradienti dei biases e dei weights
        // Inizializzo a 0 tutti i gradienti
        final ArrayList<ArrayList<Double>> biasesGradients = this.getZeroBiases();
        ArrayList<ArrayList<ArrayList<Double>>> weightsGradients = this.getZeroWeights();

        // Imposto i gradienti dei Biases dell'ultimo layer
        final int lastLayer = this.numberLayers - 1;
        // Seconda equazione fondamentale. Il gradiente del Bias di un neurone è uguale a: il valore Delta di quel neurone
        biasesGradients.set(lastLayer, deltas);

        // Imposto i gradienti dei Weights dell'ultimo layer
        // Terza equazione fondamentale. Il gradiente di un weight è uguale a: attivazione del neurone del livello precedente * delta del neurone del livello corrente
        final ArrayList<Double> activationsAtPenultimateLayer = activations.get(activations.size() - 2);
        weightsGradients = this.updateAndGetWeightsGradients(lastLayer, weightsGradients, deltas, activationsAtPenultimateLayer);

        // Calcolo di tutti i gradienti dei biases e dei weights dal penultimo livello in giù
        return this.updateAndGetBiasesAndWeightsGradientsFromPenultimateLevel(biasesGradients, weightsGradients, zs, deltas, activations);
    }

    /**
     *
     * @return Tutti i valori z e tutti i valori di attivazione di ogni neurone, livello per livello.
     * Il tipo di ritorno Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<Double>>>: elemento1 contiene i valori zs ed elemento2 contiene i valori di attivazione
     * ArrayList<ArrayList<Double>: Possiamo riassumerlo nel seguente modo: [ layer1[valore z del primo neurone, valore z del secondo neurone, ... ], layer2[valore z del primo neurone, valore z del secondo neurone, ... ], layer3[ ... ], ..., layernN[ ... ] ]
     */
    private Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<Double>>> feedforwardAndGetAllZsAndActivations(final ArrayList<Double> data) {
        // Lista che memorizza tutti i valori z, layer per layer
        final ArrayList<ArrayList<Double>> zs = new ArrayList<>();
        // Lista che memorizza tutte le attivazioni, layer per layer
        final ArrayList<ArrayList<Double>> activations = new ArrayList<>();
        ArrayList<Double> activation = data;
        activations.add(activation);

        // Faccio un feedforward memorizzando ogni z ed ogni attivazione di ogni neurone
        for (int layer = 0; layer < this.numberLayers; layer++) {
            final ArrayList<Double> neuronsZsOfLayer = new ArrayList<>();
            final ArrayList<Double> neuronsActivationsOfLayer = new ArrayList<>();
            final int numberNeurons = this.getNumberNeurons(layer);
            for (int neuron = 0; neuron < numberNeurons; neuron++) {
                final Double bias = this.getBias(layer, neuron);
                final ArrayList<Double> neuronWeights = this.getNeuronWeights(layer, neuron);

                final double z = this.computeScalarProduct(activation, neuronWeights) + bias;
                neuronsZsOfLayer.add(z);
                neuronsActivationsOfLayer.add(this.activationFunction.computeActivation(z));
            }
            zs.add(neuronsZsOfLayer);
            activations.add(neuronsActivationsOfLayer);
            activation = neuronsActivationsOfLayer;
        }

        return new Pair<>(zs, activations);
    }

    /**
     * Prima equazione fondamentale: il calcolo di Delta dell'ultimo layer. Delta è uguale a:  la derivata della funzione di costo * la derivata della funzione di attivazione
     * @param y: Labels corrette
     * @param neuronsZs: valori z dei neuroni dell'ultimo layer
     * @param neuronsActivations: valori di attivazione dei neuroni dell'ultimo layer
     * @return i valori Delta, di ogni neurone, dell'ultimo layer
     */
    private ArrayList<Double> getDeltaOfLastLayer(ArrayList<Double> y, final ArrayList<Double> neuronsZs, final ArrayList<Double> neuronsActivations) {
        ArrayList<Double> delta = new ArrayList<>();
        // Prima equazione fondamentale: Calcoliamo delta
        final int numberOutputNeurons = y.size();
        for (int neuron = 0; neuron < numberOutputNeurons; neuron++) {
            final double activationFunctionDerivative = this.activationFunction.computeDerivative(neuronsZs.get(neuron));
            final double costFunctionDerivative = this.costFunction.computeDerivative(neuronsActivations.get(neuron), y.get(neuron));
            // Delta è uguale a: la derivata della funzione di costo * la derivata della funzione di attivazione
            delta.add(activationFunctionDerivative * costFunctionDerivative);
        }

        return delta;
    }

    /**
     * Terza equazione fondamentale: il calcolo dei gradienti dei weights.
     * Il gradiente di un weight è uguale a: attivazione del neurone del livello precedente * delta del neurone del livello corrente
     * @param layer layer su cui bisogna aggiornare i weights
     * @param weightsGradients contenitore di tutti i weights di tutti i layers
     * @param deltas valori delta dei neuroni del livello corrente
     * @param activationsAtPreviousLayer attivazioni dei neuroni del livello precedente
     * @return contenitore di tutti i weights di tutti i layers aggiornato con i gradienti dei weights del livello corrente
     */
    private ArrayList<ArrayList<ArrayList<Double>>> updateAndGetWeightsGradients(final int layer,
                                                                                 final ArrayList<ArrayList<ArrayList<Double>>> weightsGradients,
                                                                                 final ArrayList<Double> deltas,
                                                                                 final ArrayList<Double> activationsAtPreviousLayer) {
        final int numberNeurons = this.getNumberNeurons(layer);

        // Terza equazione fondementale: Calcoliamo i gradienti dei weights
        for (int neuron = 0; neuron < numberNeurons; neuron++) {
            final int numberActivations = activationsAtPreviousLayer.size();
            final ArrayList<Double> neuronsWeightsGradients = new ArrayList<>();
            for (int activation = 0; activation < numberActivations; activation++) {
                neuronsWeightsGradients.add(deltas.get(neuron) * activationsAtPreviousLayer.get(activation));
            }
            weightsGradients.get(layer).set(neuron, neuronsWeightsGradients);
        }

        return weightsGradients;
    }

    /**
     * Funzione che calcola tutti i gradienti dei biases e dei weights dal penultimo livello in giù
     * Quarta equazione fondamentale: Il calcolo dei nuovi valori di delta
     * Il nuovo valore di delta è uguale a: prodotto scalere dei valori delta con i pesi che collegano il neurone con i neuroni del livello successivo
     * @param biasesGradients gradienti dei biases
     * @param weightsGradients gradienti dei weights
     * @param zs valori z di tutti i neuroni, livello per livello
     * @param deltas valori delta
     * @param activations attivazioni di tutti i neuroni, livello per livello
     * @return i gradienti dei biases e dei weights
     * Il tipo di ritorno Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>>: elemento1 contiene i valori dei gradienti dei biases ed elemento2 contiene i valori dei gradienti dei weights
     */
    private Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>> updateAndGetBiasesAndWeightsGradientsFromPenultimateLevel(final ArrayList<ArrayList<Double>> biasesGradients,
                                                                                                                                                  ArrayList<ArrayList<ArrayList<Double>>> weightsGradients,
                                                                                                                                                  final ArrayList<ArrayList<Double>> zs,
                                                                                                                                                  ArrayList<Double> deltas,
                                                                                                                                                  final ArrayList<ArrayList<Double>> activations) {
        final int penultimateLayer = this.numberLayers - 2;
        for (int layer = penultimateLayer; layer >= 0; layer--) {
            final ArrayList<Double> neuronsZs = zs.get(layer);

            // Calcolo tutte le derivate delle funzioni di attivazione del layer
            final ArrayList<Double> activationFunctionDerivatives = new ArrayList<>(
                    neuronsZs.stream().map(neuronsZ -> this.activationFunction.computeDerivative(neuronsZ)).toList()
            );

            // Quarta equazione fondamentale: Calcoliamo i nuovi delta
            // Il nuovo valore di delta è uguale a: prodotto scalere dei valori delta con i pesi che collegano il neurone con i neuroni del livello successivo
            final ArrayList<Double> newDeltas = new ArrayList<>();
            final int numberNeurons = this.getNumberNeurons(layer);
            for (int neuron = 0; neuron < numberNeurons; neuron++) {
                // Pesi che collegano il neurone con i neuroni del livello successivo
                final ArrayList<Double> neuronWeights = new ArrayList<>();
                final int numberNeuronsAtNextLevel = this.getNumberNeurons(layer + 1);

                // Ottengo i pesi che collegano il neurone con i neuroni del livello successivo
                for (int neuronAtNextLevel = 0; neuronAtNextLevel < numberNeuronsAtNextLevel; neuronAtNextLevel++) {
                    neuronWeights.add(this.getNeuronWeights(layer + 1, neuronAtNextLevel).get(neuron));
                }
                newDeltas.add(computeScalarProduct(deltas, neuronWeights) * activationFunctionDerivatives.get(neuron));
            }
            deltas = newDeltas;

            biasesGradients.set(layer, deltas);
            final ArrayList<Double> activationsOfPreviousLayer = activations.get(layer);
            weightsGradients = this.updateAndGetWeightsGradients(layer, weightsGradients, deltas, activationsOfPreviousLayer);
        }

        return new Pair<>(biasesGradients, weightsGradients);
    }

    private int getNumberNeurons(final int layer) {
        return this.weights.get(layer).size();
    }

    private Double getBias(final int layer, final int neuron) {
        return this.biases.get(layer).get(neuron);
    }

    private ArrayList<ArrayList<Double>> getNeurons(final int layer) {
        return this.weights.get(layer);
    }

    private ArrayList<Double> getNeuronWeights(final int layer, final int neuron) {
        return this.weights.get(layer).get(neuron);
    }

    private Double getWeight(final int layer, final int neuron, final int weight) {
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

    private int evaluate(final ArrayList<Data> testData) {
        final ArrayList<Pair<Integer, Integer>> predictionsAndLabels = new ArrayList<>();

        for (Data data : testData) {
            final int prediction = this.getIndexOfMaxValue(this.feedforward(data.geDataToDouble()));
            final int label = this.getIndexOfMaxValue(data.getLabelToDouble());
            predictionsAndLabels.add(new Pair<>(prediction, label));
        }

        return (int)predictionsAndLabels.stream()
                .filter(prectionAndLabel -> prectionAndLabel.getElement1().equals(prectionAndLabel.getElement2()))
                .count();
    }

    private int getIndexOfMaxValue(final ArrayList<Double> array) {
        int maxAt = 0;
        for (int i = 0; i < array.size(); i++) {
            maxAt = array.get(i) > array.get(maxAt) ? i : maxAt;
        }

        return maxAt;
    }
}
