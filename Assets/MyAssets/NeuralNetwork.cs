using System;

namespace NeuralNetwork {
    public class Layer {
        public Layer(int size) {
            NeuronNum = size;
            Bias = new Vector(size);
            Values = new Vector(size);
            Activation = new Vector(size);
        }
        public Layer(int size, Layer previousLayer) : this(size) { // [to, from]
            Weights = new Matrix(size, previousLayer.NeuronNum);
        }

        public int NeuronNum { get; }
        public Vector Bias { get; private set; }
        public Matrix Weights { get; private set; }

        public Vector Inputs { get; private set; }
        public Vector Values { get; private set; }
        public Vector Activation { get; private set; }

        public virtual Vector Forward(Vector input) {
            Inputs = input;
            Values = Weights * input + Bias;
            Activation = Values.Map(Functions.Sigmoid);
            return Values;
        }
    }

    class InputLayer:Layer {
        public InputLayer(int size) : base(size) { }

        public override Vector Forward(Vector input) {
            Activation = input;
            return input;
        }
    }

    public class NeuralNetwork {
        public NeuralNetwork(int[] layers) {
            LayerAmount = layers.Length;
            Layers = new Layer[LayerAmount];
            LayerLength = layers;

            Layers[0] = new InputLayer(layers[0]);
            for (int i = 1; i < LayerAmount; i++) {
                Layers[i] = new Layer(layers[i], Layers[i - 1]);
            }
        }

        public int LayerAmount { get; }
        public int[] LayerLength { get; }
        public Layer[] Layers { get; }

        public Matrix Calculate(Matrix input) {
            Layers[0].Forward(input);
            for (int i = 1; i < LayerAmount; i++) {
                Layers[i].Forward(Layers[i-1].Activation);
            }

            return Layers[LayerAmount - 1];
        }
    }
}