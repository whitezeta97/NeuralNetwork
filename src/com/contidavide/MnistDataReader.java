package com.contidavide;

import java.io.*;
import java.util.ArrayList;

public class MnistDataReader {

    public ArrayList<Sample> readData(String dataFilePath, String labelFilePath) throws IOException {
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        assert numberOfItems == numberOfLabels;

        final ArrayList<Sample> samples = new ArrayList<>();
        for(int i = 0; i < numberOfItems; i++) {
            final int number = labelInputStream.readUnsignedByte();
            final ArrayList<Integer> label = new ArrayList<>();
            for (int j = 0; j < 10; j++) {
                label.add(j == number ? 1 : 0);
            }

            final ArrayList<Integer> notNormalizedData = new ArrayList<>();
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    notNormalizedData.add(dataInputStream.readUnsignedByte());
                }
            }
            samples.add(new SampleImpl(notNormalizedData, label));
        }

        dataInputStream.close();
        labelInputStream.close();
        return samples;
    }
}
