package com.contidavide.optimizer;

import com.contidavide.NeuralNetworkImpl;
import com.contidavide.Pair;
import com.contidavide.Sample;
import com.contidavide.Utils;
import com.contidavide.activationfunctions.ActivationFunction;
import com.contidavide.costfunctions.CostFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Optional;

public abstract class OptimezerImpl extends NeuralNetworkImpl implements Optimizer {

    public OptimezerImpl(final ArrayList<Integer> numberNeuronsForLayer,
                  final ArrayList<ActivationFunction> activationsFunction,
                  final CostFunction costFunction) {
        super(numberNeuronsForLayer, activationsFunction, costFunction);
    }

    @Override
    public void train(final ArrayList<Sample> trainData, final Optional<ArrayList<Sample>> testData,
                      final int numberEpochs, final int miniBatchLength, final double learningRate) {
        for (int epoch = 0; epoch < numberEpochs; epoch++) {
            Collections.shuffle(trainData);
            final ArrayList<ArrayList<Sample>> miniBatches = getMiniBatches(trainData, miniBatchLength);
            for (ArrayList<Sample> miniBatch : miniBatches) {
                this.computeGradientsAndUpdateBiasesAndWeights(miniBatch, learningRate);
            }

            if (testData.isPresent()) {
                System.out.println(String.format("Epoch %s complete: %s / %s", epoch, this.evaluate(testData.get()), testData.get().size()));
            } else {
                System.out.println(String.format("Epoch %s complete", epoch));
            }
        }
    }

    /**
     * Calcola la somma di tutti i gradienti dei biases e dei weights sui dati del minibatch ed infine aggiorna i pesi dei biases e dei weights
     * @param miniBatch
     * @param learningRate
     */
    private void computeGradientsAndUpdateBiasesAndWeights(final ArrayList<Sample> miniBatch, final double learningRate) {
        // Calcolo la somma di tutti i gradienti dei biases e dei weights sui dati del minibatch
        final Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>> biasesAndWeightsGradients = this.computeBiasesAndWeightsGradients(miniBatch);
        final ArrayList<ArrayList<Double>> biasesGradients = biasesAndWeightsGradients.getElement1();
        final ArrayList<ArrayList<ArrayList<Double>>> weightsGradients = biasesAndWeightsGradients.getElement2();
        // Aggiorno i pesi dei biases e dei weights
        this.updateBiasesAndWeights(biasesGradients, weightsGradients, miniBatch.size(), learningRate);
    }

    /**
     * Calcola la somma di tutti i gradienti dei biases e dei weights sui dati del minibatch.
     * Per ogni dato nel minibatch si effettua la backpropagation per calcolare i gradienti. Successivamente i gradienti ottenuti vengono sommati ai gradienti totali
     * @param miniBatch
     * @return la somma di tutti i gradienti dei biases e dei weights effettuata sul minibatch
     * Il tipo di ritorno Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>>: elemento1 contiene i valori dei gradienti dei biases ed elemento2 contiene i valori dei gradienti dei weights
     */
    private Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>> computeBiasesAndWeightsGradients(final ArrayList<Sample> miniBatch) {

        final ArrayList<ArrayList<Double>> totalBiasesGradients = this.getZeroBiases();
        final ArrayList<ArrayList<ArrayList<Double>>> totalWeightsGradients = this.getZeroWeights();

        for (Sample sample : miniBatch) {
            Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>> biasesAndWeightsGradients = this.backpropagation(sample);
            final ArrayList<ArrayList<Double>> biasesGradientsOfBackpropagation = biasesAndWeightsGradients.getElement1();
            final ArrayList<ArrayList<ArrayList<Double>>> weightsGradientsOfBackpropagation = biasesAndWeightsGradients.getElement2();

            // Update gradients of Biases and Weights
            for (int layer = 0; layer < super.numberLayers; layer++) {
                final int numberNeurons = super.getNumberNeurons(layer);
                for (int neuron = 0; neuron < numberNeurons; neuron++) {
                    // Update gradients of Biases
                    final Double totalBiasGradient = totalBiasesGradients.get(layer).get(neuron);
                    final Double biasGradientOfBackPropagation = biasesGradientsOfBackpropagation.get(layer).get(neuron);
                    totalBiasesGradients.get(layer).set(neuron, totalBiasGradient + biasGradientOfBackPropagation);

                    // Update gradients of Weights
                    final int numberWeights = super.getNeuronWeights(layer, neuron).size();
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

    /**
     * Algoritmo di backpropagation
     *
     * @return tutti i gradienti dei biases e dei weights.
     * Il tipo di ritorno Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>>: elemento1 contiene i valori dei gradienti dei biases ed elemento2 contiene i valori dei gradienti dei weights
     * */
    private Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<ArrayList<Double>>>> backpropagation(final Sample sample) {
        // Calcolo di tutti i valori z e tutti i valori di attivazione di ogni neurone, livello per livello
        final Pair<ArrayList<ArrayList<Double>>, ArrayList<ArrayList<Double>>> zsAndActivations =
                this.feedforwardAndGetAllZsAndActivations(sample.getNormalizedData());
        final ArrayList<ArrayList<Double>> zs = zsAndActivations.getElement1();
        final ArrayList<ArrayList<Double>> activations = zsAndActivations.getElement2();

        // Prima equazione fondamentale: Calcolo tutti i valori Delta, di ogni neurone, dell'ultimo layer
        final ArrayList<Double> zsOfLastLayer = zs.get(zs.size() - 1);
        final ArrayList<Double> activationsOfLastLayer = activations.get(activations.size() - 1);
        ArrayList<Double> deltas = this.getDeltaOfLastLayer(sample.getLabelToDouble(), zsOfLastLayer, activationsOfLastLayer);

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
        for (int layer = 0; layer < super.numberLayers; layer++) {
            final ActivationFunction activationFunction = super.activationsFunction.get(layer);
            final ArrayList<Double> neuronsZsOfLayer = new ArrayList<>();
            final ArrayList<Double> neuronsActivationsOfLayer = new ArrayList<>();
            final int numberNeurons = super.getNumberNeurons(layer);
            for (int neuron = 0; neuron < numberNeurons; neuron++) {
                final Double bias = super.getBias(layer, neuron);
                final ArrayList<Double> neuronWeights = super.getNeuronWeights(layer, neuron);

                final double z = super.computeScalarProduct(activation, neuronWeights) + bias;
                neuronsZsOfLayer.add(z);
                neuronsActivationsOfLayer.add(activationFunction.computeActivation(z));
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
        final ActivationFunction activationFunction = super.activationsFunction.get(super.activationsFunction.size() - 1);
        ArrayList<Double> delta = new ArrayList<>();
        // Prima equazione fondamentale: Calcoliamo delta
        final int numberOutputNeurons = y.size();
        for (int neuron = 0; neuron < numberOutputNeurons; neuron++) {
            final double activationFunctionDerivative = activationFunction.computeDerivative(neuronsZs.get(neuron));
            final double costFunctionDerivative = super.costFunction.computeDerivative(neuronsActivations.get(neuron), y.get(neuron));
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
        final int numberNeurons = super.getNumberNeurons(layer);

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
        final int penultimateLayer = super.numberLayers - 2;
        for (int layer = penultimateLayer; layer >= 0; layer--) {
            final ArrayList<Double> neuronsZs = zs.get(layer);

            // Calcolo tutte le derivate delle funzioni di attivazione del layer
            final ActivationFunction activationFunction = super.activationsFunction.get(layer);
            final ArrayList<Double> activationFunctionDerivatives = new ArrayList<>(
                    neuronsZs.stream().map(neuronsZ -> activationFunction.computeDerivative(neuronsZ)).toList()
            );

            // Quarta equazione fondamentale: Calcoliamo i nuovi delta
            // Il nuovo valore di delta è uguale a: prodotto scalere dei valori delta con i pesi che collegano il neurone con i neuroni del livello successivo
            final ArrayList<Double> newDeltas = new ArrayList<>();
            final int numberNeurons = super.getNumberNeurons(layer);
            for (int neuron = 0; neuron < numberNeurons; neuron++) {
                // Pesi che collegano il neurone con i neuroni del livello successivo
                final ArrayList<Double> neuronWeights = new ArrayList<>();
                final int numberNeuronsAtNextLevel = super.getNumberNeurons(layer + 1);

                // Ottengo i pesi che collegano il neurone con i neuroni del livello successivo
                for (int neuronAtNextLevel = 0; neuronAtNextLevel < numberNeuronsAtNextLevel; neuronAtNextLevel++) {
                    neuronWeights.add(super.getNeuronWeights(layer + 1, neuronAtNextLevel).get(neuron));
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
        for (int layer = 0; layer < super.numberLayers; layer++) {
            final int numberNeurons = super.getNumberNeurons(layer);
            for (int neuron = 0; neuron < numberNeurons; neuron++) {
                // Update Bias
                final double biasGradient = biasesGradients.get(layer).get(neuron);
                final Double currentBias = super.getBias(layer, neuron);
                final Double updatedBias = currentBias - (learningRate / miniBatchSize) * biasGradient;
                super.biases.get(layer).set(neuron, updatedBias);

                final int numberWeights = super.getNeuronWeights(layer, neuron).size();
                for (int weight = 0; weight < numberWeights; weight++) {
                    // Update weight
                    final Double weightGradient = weightsGradients.get(layer).get(neuron).get(weight);
                    final Double currentWeight = super.getWeight(layer, neuron, weight);
                    final Double updatedWeight = currentWeight - (learningRate / miniBatchSize) * weightGradient;
                    super.weights.get(layer).get(neuron).set(weight, updatedWeight);
                }
            }
        }
    }

    private ArrayList<ArrayList<Sample>> getMiniBatches(final ArrayList<Sample> trainData, final int miniBatchLength) {
        final ArrayList<ArrayList<Sample>> miniBatches = new ArrayList<>();
        ArrayList<Sample> miniBatch = new ArrayList<>();
        int numberElementsInMiniBatch = 0;

        for (Sample sample : trainData) {
            miniBatch.add(sample);
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

    private ArrayList<ArrayList<Double>> getZeroBiases() {
        final ArrayList<ArrayList<Double>> zeroBiases = new ArrayList<>();
        super.biases.forEach(layerBiases -> zeroBiases.add(new ArrayList<>(Collections.nCopies(layerBiases.size(), 0.0))));

        return zeroBiases;
    }

    private ArrayList<ArrayList<ArrayList<Double>>> getZeroWeights() {
        final ArrayList<ArrayList<ArrayList<Double>>> zeroWeights = new ArrayList<>();
        super.weights.forEach(layerWeights -> {
            final ArrayList<ArrayList<Double>> weights = new ArrayList<>();
            layerWeights.forEach(neuronsWeights -> weights.add(new ArrayList<>(Collections.nCopies(neuronsWeights.size(), 0.0))));
            zeroWeights.add(weights);
        });

        return zeroWeights;
    }

    private int evaluate(final ArrayList<Sample> testData) {
        final ArrayList<Pair<Integer, Integer>> predictionsAndLabels = new ArrayList<>();

        for (Sample sample : testData) {
            final int prediction = Utils.getIndexOfMaxValue(super.feedforward(sample.getNormalizedData()));
            final int label = Utils.getIndexOfMaxValue(sample.getLabelToDouble());
            predictionsAndLabels.add(new Pair<>(prediction, label));
        }

        return (int)predictionsAndLabels.stream()
                .filter(prectionAndLabel -> prectionAndLabel.getElement1().equals(prectionAndLabel.getElement2()))
                .count();
    }
}
