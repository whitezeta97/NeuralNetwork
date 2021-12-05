package com.contidavide;

import java.util.ArrayList;

public interface Sample {
    void setStandardization();

    ArrayList<Integer> getNotNormalizedData();

    ArrayList<Double> getNormalizedData();

    ArrayList<Integer> getLabel();

    ArrayList<Double> getLabelToDouble();
}
