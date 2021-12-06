package com.contidavide;

import java.util.ArrayList;

public class SampleImpl implements Sample {
    private ArrayList<Integer> label;
    private ArrayList<Integer> notNormalizedData;
    private ArrayList<Double> normalizedData;


    SampleImpl(final ArrayList<Integer> notNormalizedData, final ArrayList<Integer> label) {
        this.notNormalizedData = notNormalizedData;
        this.label = label;
        this.normalizedData = new ArrayList<>();
    }

    @Override
    public void setStandardization() {
        double media = 0;
        for (Integer data : this.notNormalizedData) {
            media += data;
        }
        media /= this.notNormalizedData.size();

        double std = 0;
        for (Integer data : this.notNormalizedData) {
            std += Math.pow(data - media, 2);
        }
        std = Math.sqrt(std / this.notNormalizedData.size());

        for (Integer data : this.notNormalizedData) {
            final Double normalizedData = (data - media) / std;
            this.normalizedData.add(normalizedData);
        }
    }

    @Override
    public ArrayList<Integer> getLabel() {
        return this.label;
    }

    @Override
    public ArrayList<Double> getLabelToDouble() {
        return new ArrayList<>(this.label.stream().mapToDouble(i -> i).boxed().toList());
    }

    @Override
    public ArrayList<Integer> getNotNormalizedData() {
        return this.notNormalizedData;
    }

    @Override
    public ArrayList<Double> getNormalizedData() {
        return this.normalizedData;
    }
}
