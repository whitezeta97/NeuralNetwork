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
package com.contidavide;

public class Pair<T, E> {
    private T element1;
    private E element2;

    Pair(final T element1, final E element2) {
        this.element1 = element1;
        this.element2 = element2;
    }

    public T getElement1() {
        return element1;
    }

    public E getElement2() {
        return element2;
    }
}