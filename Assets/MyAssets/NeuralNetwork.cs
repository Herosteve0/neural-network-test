using System;

public class Layer {
    public Layer(int size) {
        NeuronNum = size;
        Bias = new Matrix(size, 1);
        Values = new Matrix(size, 1);
        Activation = new Matrix(size, 1);
    }
    public Layer(int size, Layer previousLayer) : this(size) { // [to, from]
        Weights = new Matrix(size, previousLayer.NeuronNum);
    }

    public int NeuronNum { get; }
    public Matrix Bias { get; private set; }
    public Matrix Weights { get; private set; }

    public Matrix Inputs { get; private set;  }
    public Matrix Values { get; private set; }
    public Matrix Activation { get; private set; }

    public virtual Matrix Forward(Matrix input) {
        Inputs = input;
        Values = Weights * input + Bias;
        Activation = Values.Map(Functions.Sigmoid);
        return Values;
    }

    public virtual Matrix Backprop(Matrix output, Matrix expected) {
        return Matrix.Zero(3);
    }
}

class InputLayer : Layer {
    public InputLayer(int size) : base(size) { }

    public override Matrix Forward(Matrix input) {
        return input;
    }
}

public class NeuralNetwork {
    public NeuralNetwork(int[] layers) {
        Layers = new Layer[layers.Length];

        Layers[0] = new InputLayer(layers[0]);
        for (int i = 1; i < layers.Length; i++) {
            Layers[i] = new Layer(layers[i], Layers[i-1]);
        }
    }

    public Layer[] Layers { get; }
}