package com.contidavide;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Optional;

public class Main {

    public static void main(String[] args) throws Exception {
        ArrayList<Sample> trainData = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        ArrayList<Sample> testData = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

        // Standardization
        trainData.forEach(data -> data.setStandardization());
        testData.forEach(data -> data.setStandardization());

        System.out.println("Neural network is building");
        final NeuralNetwork neuralNetwork = buildNeuralNetwork();
        System.out.println("Neural network is build");

        String input = "";
        while (!"3".equals(input)) {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
            System.out.println("Enter 1 for prediction");
            System.out.println("Enter 2 for train");
            System.out.println("Enter 3 for exit");

            input = bufferedReader.readLine();
            if ("1".equals(input)) {
                prediction(neuralNetwork, testData);
            } else if ("2".equals(input)) {
                System.out.println("Train start");
                neuralNetwork.trainWithStochasticGradientDescent(trainData, Optional.of(testData), 10, 10, 3.0);
            }
        }
    }

    private static NeuralNetwork buildNeuralNetwork() throws Exception {
        return new NeuralNetworkBuildImpl()
                .addLayer(784)
                .addLayer(30)
                .addLayer(10)
                .addMeanSquareErrorCostFunction()
                .addSigmoidActivationFunction()
                .build();
    }

    private static void prediction(final NeuralNetwork neuralNetwork, final ArrayList<Sample> testData) throws Exception {
        System.out.println("Number of sample (0 - 9999)");
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        final String input = bufferedReader.readLine();

        final Sample sample = testData.get(Integer.parseInt(input));
        System.out.println("Image is:");
        printSample(sample);

        final ArrayList<Double> outputs = neuralNetwork.feedforward(sample.getNormalizedData());
        System.out.println("Output of neural network: " + getIndexOfMaxValue(outputs));
        System.out.println("Desired Output: " + getIndexOfMaxValue(sample.getLabelToDouble()));
    }


    private static void printSample(final Sample sample) {
        int count = 0;
        for (int r = 0; r < 784; r++ ) {
            System.out.print(sample.getNotNormalizedData().get(r) + " ");
            count++;
            if (count == 28) {
                System.out.println("");
                count = 0;
            }
        }
    }

    private static int getIndexOfMaxValue(final ArrayList<Double> array) {
        int maxAt = 0;
        for (int i = 0; i < array.size(); i++) {
            maxAt = array.get(i) > array.get(maxAt) ? i : maxAt;
        }

        return maxAt;
    }
}
