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