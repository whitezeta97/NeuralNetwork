package com.contidavide;

import java.util.ArrayList;
import java.util.Optional;

public interface NeuralNetwork {
    ArrayList<Double> feedforward(ArrayList<Double> inputNeurons);
}
