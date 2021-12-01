/*******************************************************************************
 * Copyright (c) 1999-2021 Iungo S.p.A.
 * All Rights Reserved.
 *
 * mailto: iungo-development AT iungo DOT com
 *
 * NOTICE:  All information contained herein is, and remains the property of
 * Iungo SpA.
 * The intellectual and technical concepts contained herein are proprietary to
 * Iungo SpA and may be covered by Italian and Foreign Patents, patents in
 * process, and are protected by trade secret or copyright law.
 * Dissemination of this information or reproduction of this material is
 * strictly forbidden unless prior written permission is obtained from
 * Iungo SpA.
 *
 */
package com.contidavide.costfunctions;

public class MeanSquareError implements CostFunction {

    @Override
    public double computeDerivative(final double activation, final double y) {
        return activation - y;
    }
}