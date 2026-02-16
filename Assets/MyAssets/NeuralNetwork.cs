using System;
using System.Diagnostics;
using UnityEngine.Rendering.Universal;

namespace NeuralNetworkSystem {
    public class Layer {
        public Layer(int size) {
            NeuronNum = size;
            Bias = Vector.Random(size, -1f, 1f);
            Values = Vector.Random(size, -1f, 1f);
            Activation = Vector.Random(size, -1f, 1f);
        }
        public Layer(int size, Layer previousLayer) : this(size) { // [to, from]
            Weights = Matrix.Random(size, previousLayer.NeuronNum, -1f, 1f);
        }

        public int NeuronNum { get; }
        public Vector Bias { get; set; }
        public Matrix Weights { get; set; }

        public Vector Inputs { get; protected set; }
        public Vector Values { get; protected set; }
        public Vector Activation { get; protected set; }

        public virtual Vector Forward(Vector input) {
            Inputs = input;
            Values = Weights * input + Bias;
            Activation = Values.Map(NeuralNetworkTrainer.Sigmoid);
            return Activation;
        }
    }

    class InputLayer : Layer {
        public InputLayer(int size) : base(size) { }

        public override Vector Forward(Vector input) {
            Activation = input;
            return input;
        }
    }

    class OutputLayer : Layer {
        public OutputLayer(int size, Layer previousLayer) : base(size, previousLayer) { }

        public override Vector Forward(Vector input) {
            base.Forward(input);
            Activation = Values.SoftMax();
            return Activation;
        }
    }

    public class NeuralNetwork {
        public NeuralNetwork(int[] layers) {
            LayerAmount = layers.Length;
            Layers = new Layer[LayerAmount];
            LayerLength = new int[LayerAmount];

            for (int i = 0; i < LayerAmount; i++) {
                LayerLength[i] = layers[i];
                if (i == 0) {
                    Layers[0] = new InputLayer(layers[i]);
                } else if (i == LayerAmount - 1) {
                    Layers[LayerAmount - 1] = new OutputLayer(layers[i], Layers[i - 1]);
                } else {
                    Layers[i] = new Layer(layers[i], Layers[i - 1]);
                }
            }
        }

        public int LayerAmount { get; }
        public int[] LayerLength { get; }
        public Layer[] Layers { get; }

        public Vector Calculate(Vector input) {
            Layers[0].Forward(input);
            for (int i = 1; i < LayerAmount; i++) {
                Layers[i].Forward(Layers[i-1].Activation);
            }

            return Layers[LayerAmount - 1].Activation;
        }
    }
}