package com.contidavide;

import java.util.ArrayList;

public class Utils {
    public static int getIndexOfMaxValue(final ArrayList<Double> array) {
        int maxAt = 0;
        for (int i = 0; i < array.size(); i++) {
            maxAt = array.get(i) > array.get(maxAt) ? i : maxAt;
        }

        return maxAt;
    }
}