using System;
using System.Numerics;

namespace NeuralNetworkSystem {
    public class Layer {
        float WeightScaler(int previousLength) {
            return UnityEngine.Mathf.Sqrt(6f / previousLength);
        }

        Func<float, float> ActivationFunction = NeuralNetworkTrainer.ReLU;

        public Layer(int size) {
            NeuronNum = size;
            Bias = Vector.Random(size, -1f, 1f);
            Values = new Vector(size);
            Activation = new Vector(size);
            Delta = new Vector(size);
        }
        public Layer(int size, Layer previousLayer) : this(size) { // [to, from]
            float value = WeightScaler(previousLayer.NeuronNum);
            Weights = Matrix.Random(size, previousLayer.NeuronNum, -value, value);
        }

        public int NeuronNum { get; }
        public Vector Bias { get; set; }
        public Matrix Weights { get; set; }

        public Vector Values { get; protected set; }
        public Vector Activation { get; protected set; }

        public Vector Delta { get; protected set; }

        public virtual void Forward(Vector input) {
            CalculateValue(input);
            Values.Map(ActivationFunction, Activation);
        }
        protected void CalculateValue(Vector input) {
            
            int simd_width = Vector<float>.Count; // 8

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
            }
        }

        public virtual void Backward(Vector PreviousValues, Vector DeltaOut) {
            int simd_width = Vector<float>.Count; // 8
            
            // Weight is transposed, so rows and columns are reversed.

            for (int row = 0; row < Weights.Columns; row++) {
                float sum = 0f;
                int offset = row * Weights.Rows;

                int col = 0;
                for (; col <= Weights.Rows - simd_width; col += simd_width) {
                    var v_weights = new Vector<float>(Weights.Data, offset + col);
                    var v_delta = new Vector<float>(Delta.Data, col);
                    sum += System.Numerics.Vector.Dot(v_weights, v_delta);
                }

                for (; col < Weights.Rows; col++) {
                    sum += Weights.Data[offset + col] * Delta.Data[col];
                }

                //DeltaOut.Data[row] = sum * NeuralNetworkTrainer.ReLUDerivative(PreviousValues.Data[row]);
                DeltaOut.Data[row] = PreviousValues.Data[row] > 0f ? sum : 0f;
            }

                //(Network.Layers[l + 1].Weights.Transpose() * Delta[l])
                //    .ElementMultiply(
                //        Network.Layers[l].Values.Map(ReLUDerivative)
                //    );
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
            CalculateValue(input);
            Values.SoftMax(Activation);
        }

        public override void Backward(Vector PreviousValues, Vector DeltaOut) {
            Vector.Sub(Activation, PreviousValues, DeltaOut);
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