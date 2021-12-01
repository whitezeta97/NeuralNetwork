package com.contidavide;

import java.util.ArrayList;
import java.util.Optional;

public class Main {

    public static void main(String[] args) throws Exception {
        ArrayList<Data> trainData = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        ArrayList<Data> testData = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

        System.out.println("Neural network is building");
        final NeuralNetwork neuralNetwork = new NeuralNetworkBuildImpl()
                .addLayer(784)
                .addLayer(30)
                .addLayer(10)
                .addMeanSquareErrorCostFunction()
                .addSigmoidActivationFunction()
                .build();
        System.out.println("Neural network is build");

//        System.out.println(neuralNetwork.feedforward(trainData.get(0).geDataToDouble()));

        neuralNetwork.trainWithStochasticGradientDescent(trainData, Optional.of(testData), 30, 10, 3.0);
    }


    private static void printMnistData(final Data data) {
        System.out.println("label: " + data.getLabel());
//        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
//            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
//                System.out.print(matrix.getValue(r, c) + " ");
//            }
//            System.out.println();
//        }

        int count = 0;
        for (int r = 0; r < 784; r++ ) {
            System.out.print(data.getData().get(r) + " ");
            count++;
            if (count == 28) {
                System.out.println("");
                count = 0;
            }
        }
    }
}
