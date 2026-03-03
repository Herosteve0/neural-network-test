using System;
using System.Diagnostics;
using System.Numerics;
using UnityEditor.Rendering;

namespace NeuralNetworkSystem {
    public class Layer {
        float WeightScaler(int previousLength) {
            return UnityEngine.Mathf.Sqrt(6f / previousLength);
        }

        Func<float, float> ActivationFunction = NeuralNetworkTrainer.ReLU;

        public Layer(int size) {
            NeuronNum = size;
            Bias = new Vector(size);
            Values = new Vector(size);
            Activation = new Vector(size);
            Delta = new Vector(size);
        }
        public Layer(int size, Layer previousLayer) : this(size) { // [to, from]
            float value = WeightScaler(previousLayer.NeuronNum);
            Weights = Matrix.Random(size, previousLayer.NeuronNum, -value, value);
            WeightsT = new Matrix(previousLayer.NeuronNum, size);
            Weights.Transpose(WeightsT);
            previousLayer.NextLayer = this;
        }

        public int NeuronNum { get; }
        public Vector Bias { get; set; }
        public Matrix Weights { get; set; }
        public Matrix WeightsT { get; set; }

        public Vector Values { get; protected set; }
        public Vector Activation { get; protected set; }

        public Vector Delta { get; protected set; }
        public Layer NextLayer { get; protected set; }

        public virtual void Forward(Vector input) {
            CalculateValue(input);
        }
        protected void CalculateValue(Vector input) {
            int simd_width = Vector<float>.Count; // 8

            // Weights * Input + Bias

            for (int row = 0; row < Weights.Rows; row++) {
                float sum = Bias[row];
                int offset = row * Weights.Columns;
                    
                int col = 0;
                for (; col <= Weights.Columns - simd_width; col += simd_width) {
                    var v_weights = new Vector<float>(Weights.Data, offset + col);
                    var v_x = new Vector<float>(input.Data, col);
                    sum += System.Numerics.Vector.Dot(v_weights, v_x);
                }

                for (; col < Weights.Columns; col++) {
                    sum += Weights.Data[offset + col] * input.Data[col];
                }

                Values[row] = sum;
                Activation[row] = ActivationFunction(sum);
            }
        }

        public virtual void BackwardOutput(Vector CorrectValues) { }
        public virtual void Backward() {
            int simd_width = Vector<float>.Count; // 8

            // Weight is transposed, so rows and columns are reversed.

            int Rows = NextLayer.WeightsT.Rows;
            int Columns = NextLayer.WeightsT.Columns;

            bool flag = false;

            //UnityEngine.Debug.Log($"NextLayer.Delta: {NextLayer.Delta.Length}, NextLayer.Weights: {NextLayer.Weights.Rows}x{NextLayer.Weights.Columns}, Delta: {Delta.Length}, NextLayer.Values: {NextLayer.Values.Length}");
            for (int row = 0; row < Rows; row++) {
                float sum = 0f;
                int offset = row * Columns;

                // Sum of N.W.T[row,...] * N.D[...]
                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v_weights = new Vector<float>(NextLayer.WeightsT.Data, offset + col);
                    var v_delta = new Vector<float>(NextLayer.Delta.Data, col);
                    sum += System.Numerics.Vector.Dot(v_weights, v_delta);
                }

                for (; col < Columns; col++) {
                    sum += NextLayer.WeightsT.Data[offset + col] * NextLayer.Delta.Data[col];
                }

                if (float.IsNaN(sum)) flag = true;

                // Sum * ReLU'(Z[row])

                Delta.Data[row] = sum * NeuralNetworkTrainer.ReLUDerivative(Values.Data[row]);
                //Delta.Data[row] = Values.Data[row] > 0f ? sum : 0f;
            }

            if (flag) UnityEngine.Debug.Log("NaN detected!");
        }
    }

    class InputLayer : Layer {
        public InputLayer(int size) : base(size) { }

        public override void Forward(Vector input) {
            Activation = input;
        }
    }

    class OutputLayer : Layer {
        public OutputLayer(int size, Layer previousLayer) : base(size, previousLayer) { }

        public override void Forward(Vector input) {
            base.Forward(input);
            Values.SoftMax(Activation);
        }

        public override void BackwardOutput(Vector CorrectValues) {
            Vector.Sub(Activation, CorrectValues, Delta);
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