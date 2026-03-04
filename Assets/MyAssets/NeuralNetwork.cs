using System;
using System.Numerics;

namespace NeuralNetworkSystem {
    public class Layer {
        float WeightScaler(int previousLength) {
            return UnityEngine.Mathf.Sqrt(6f / previousLength);
        }

        Func<float, float> ActivationFunction = NeuralNetworkTrainer.ReLU;
        Func<float, float> ActivationFunctionDerivative = NeuralNetworkTrainer.ReLUDerivative;

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
            PreviousLayer = previousLayer;
            previousLayer.NextLayer = this;
        }

        public int NeuronNum { get; }
        public Vector Bias { get; set; }
        public Matrix Weights { get; set; }
        public Matrix WeightsT { get; set; }

        public Vector Values { get; protected set; }
        public Vector Activation { get; protected set; }

        public Vector Delta { get; protected set; }

        public Layer PreviousLayer { get; protected set; }
        public Layer NextLayer { get; protected set; }






        Func<Vector, Vector> InputNormalizationFunc;
        Func<Vector, Vector> ForwardCalculationFunc;
        Func<Vector, Vector> ForwardOutputFunc;

        Func<Vector, Vector> BackwardDeltaFunc;
        Func<Vector, Vector> BackwardWeightsFunc;
        Func<Vector, Vector> BackwardsBiasFunc;

        Func<Vector, Vector> AdjustWeightsFunc;
        Func<Vector, Vector> AdjustBiasFunc;


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

        public virtual void Backward(Vector CorrectValues) { // input only used in output layer
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

                Delta.Data[row] = sum * ActivationFunctionDerivative(Values.Data[row]);
                //Delta.Data[row] = Values.Data[row] > 0f ? sum : 0f;
            }

            if (flag) UnityEngine.Debug.Log("NaN detected!");
        }
        public virtual void BackwardWeights(ref Matrix WeightDelta) {
            int simd_width = Vector<float>.Count; // 8

            int Rows = Delta.Length;
            int Columns = PreviousLayer.Activation.Length;


            for (int row = 0; row < Rows; row++) {
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v = new Vector<float>(PreviousLayer.Activation.Data, col);
                    var v_weight = new Vector<float>(WeightDelta.Data, offset + col);

                    v = Delta.Data[row] * v;
                    (v + v_weight).CopyTo(WeightDelta.Data, offset + col);
                }

                for (; col < Columns; col++) {
                    WeightDelta.Data[offset + col] += Delta.Data[row] * PreviousLayer.Activation.Data[col];
                }
            }
        }
        public virtual void BackwardBias(ref Vector BiasDelta) {
            int simd_width = Vector<float>.Count; // 8

            int Columns = Delta.Length;

            int col = 0;
            for (; col <= Columns - simd_width; col += simd_width) {
                var v = new Vector<float>(Delta.Data, col);
                var v_bias = new Vector<float>(BiasDelta.Data, col);
                (v + v_bias).CopyTo(BiasDelta.Data, col);
            }

            for (; col < Columns; col++) {
                BiasDelta.Data[col] += Delta.Data[col];
            }
        }

        public virtual void AdjustWeight(ref Matrix WeightsDelta, float scale) {
            if (Weights.Rows != WeightsDelta.Rows) throw new Exception("Weights and WeightsDelta don't have matching Rows!");
            if (Weights.Columns != WeightsDelta.Columns) throw new Exception("Weights and WeightsDelta don't have matching Columns!");

            int simd_width = Vector<float>.Count;
            int length = Weights.Rows * Weights.Columns;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = new Vector<float>(WeightsDelta.Data, i);
                v_delta *= scale;
                var v = new Vector<float>(Weights.Data, i);
                (v - v_delta).CopyTo(Weights.Data, i);
            }
            for (; i < length; i++) {
                Weights.Data[i] -= WeightsDelta.Data[i] * scale;
            }

            for (int row = 0; row < WeightsT.Rows; row++) {
                for (int col = 0; col < WeightsT.Columns; col++) {
                    WeightsT[row, col] -= WeightsDelta[col, row] * scale;
                }
            }
        }
        public virtual void AdjustBias(ref Vector BiasDelta, float scale) {
            if (Bias.Length != BiasDelta.Length) throw new Exception("Bias and BiasDelta don't have matching Lengths!");

            int simd_width = Vector<float>.Count;
            int length = Bias.Length;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = new Vector<float>(BiasDelta.Data, i);
                v_delta *= scale;
                var v = new Vector<float>(Bias.Data, i);
                (v - v_delta).CopyTo(Bias.Data, i);
            }
            for (; i < length; i++) {
                Bias.Data[i] -= BiasDelta.Data[i] * scale;
            }
        }
    }

    class InputLayer : Layer {
        public InputLayer(int size) : base(size) { }

        public override void Forward(Vector input) {
            Activation.Data = InputFunctions.NormalizeMedian(input.Data);
        }
    }

    class OutputLayer : Layer {
        public OutputLayer(int size, Layer previousLayer) : base(size, previousLayer) { }

        public override void Forward(Vector input) {
            base.Forward(input);
            Activation.Data = OutputFunctions.SoftMax(Values.Data);
        }

        public override void Backward(Vector CorrectValues) {
            int simd_width = Vector<float>.Count;

            int i = 0;
            for (; i <= Delta.Length - simd_width; i += simd_width) {
                var v_a = new Vector<float>(Activation.Data, i);
                var v_b = new Vector<float>(CorrectValues.Data, i);
                (v_a - v_b).CopyTo(Delta.Data, i);
            }
            for (; i < Delta.Length; i++) {
                Delta.Data[i] = Activation[i] - CorrectValues[i];
            }
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