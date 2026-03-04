using System;
using System.Numerics;

namespace NeuralNetworkSystem {
    public enum ActivationFunctionsTypes {
        Sigmoid,
        ReLU
    }
    public abstract class ActivationFunctions {
        // Sigmoid heavily punishes bad Neurons, while heavily rewarding good Neurons.
        public static float Sigmoid(float value) {
            float e = (float)Math.Exp(-value);
            return - e / (e + 1);
        }
        public static float SigmoidDerivative(float value) {
            float e = (float)Math.Exp(-value);
            return e / ((1 + e) * (1 + e));
        }

        // ReLU heavily rewards good Neurons while essentially ignoring bad ones.
        public static float ReLU(float value) {
            return value >= 0 ? value : 0;
        }
        public static float ReLUDerivative(float value) {
            return value >= 0 ? 1 : 0;
        }
    }

    public enum InputFunctionsType {
        Normalize,
        NormalizeMeadian
    }
    public abstract class InputFunctions {
        public static float[] Normalize(float[] input) {
            int simd_width = Vector<float>.Count;

            float[] r = new float[input.Length];

            float scale = 1 / 255f;

            int i = 0;
            for (; i <= input.Length - simd_width; i += simd_width) {
                var v = new Vector<float>(input, i);
                (v * scale).CopyTo(r, i);
            }
            for (; i < input.Length; i++) {
                r[i] *= scale;
            }

            return r;
        }

        public static float[] NormalizeMedian(float[] input) {
            float mean = 0.1307f;
            float std = 0.3081f;

            float[] r = new float[input.Length];
            for (int i = 0; i < input.Length; i++) {
                r[i] = (input[i] - mean) / std;
            }
            return r;
        }
    }

    public enum OutputFunctionsType {
        SoftMax
    }
    public abstract class OutputFunctions {
        /*

        SoftMax is a function that takes a Vector, or in general an array of data, and returns
        a Vector that has the probability distribution of these values.
        The sum of this Vector will always be 1.

        Assuming N = the length of the final layer
        We have to find the following:
        Max, which is O(N)
        Sum, which is O(N)
        Result, which again, is O(N)

        So we end up having O(3N) time complexity, which simplifies to O(N)

        */
        public static float[] SoftMax(float[] output) {
            int length = output.Length;
            float[] r = new float[length];

            float max = output[0];
            for (int i = 1; i < length; i++) { // N loops
                if (max < output[i]) max = output[i];
            }

            float sum = 0f;
            for (int i = 0; i < length; i++) { // N loops
                float e = (float)Math.Exp(output[i] - max);
                r[i] = e;
                sum += e;
            }

            for (int i = 0; i < length; i++) { // N loops
                r[i] /= sum;
            }

            return r;
        }
    }
    
    public enum LossFunctionsType {
        Mean,
        SoftMax
    }
    public abstract class LossFunctions {
        public static float Mean(float[] V, int label) {
            float a = V[label] - 1;
            return a * a;
        }

        public static float SoftMax(Vector V, int label) {
            return -UnityEngine.Mathf.Log(V[label]);
        }
    }

    public enum ForwardFunctionsTypes {
        Simple,
        Parallel,
        SSE,
        AVX
    }
    public enum BackwardFunctionsTypes {
        Simple,
        Parallel,
        SSE,
        AVX
    }
    public enum BackwardWeightsFunctionsTypes {
        Simple,
        Parallel,
        SSE,
        AVX
    }
    public enum BackwardBiasFunctionsTypes {
        Simple,
        Parallel,
        SSE,
        AVX
    }
    public enum AdjustdWeightsFunctionsTypes {
        Simple,
        Parallel,
        SSE,
        AVX
    }
    public enum AdjustBiasFunctionsTypes {
        Simple,
        Parallel,
        SSE,
        AVX
    }

    public abstract class EducationalFunctions {
        public static void CalculateValue(Layer layer, Func<float, float> ActivationFunc, Vector input) {

            /*
            
            Z[L] = Weights[L] * A[L-1] + Bias[L]

            Reminder:
            A[L] is the activation of the Layer
            Z[L] is the pre-activation of the Layer

            so in other words, this expression can be described as

            My values = My weight * Previous layer's activation (input) + My bias

            */

            layer.Values.Data = (layer.Weights * input + layer.Bias).Data;

            /*
            
            A[L] = σ(Z[L])

            σ(x) being the Activation function, which in our case is ActivationFunc

            */
            layer.Activation.Data = layer.Values.Map(ActivationFunc).Data;
        }

        public static void Backward(Layer layer, Func<float, float> ActivationFuncDer) {

            /*
            
            D[L] = (W[L+1].T * D[L+1]) ⊙ σ'(Z[L])
            
            In the Output Layer, the Delta is calculate by subtracting the Activation with the Expected Values (A[Last Layer] - Y)

            Delta is what we call "Gradient Descend", which tells the parameters of our network how to adjust, in order to correct themselves.
            In Neural Networks, the act of calculating more than one Delta (in other words, the network has Hidden Layers), is called "Backpropagation"
            The delta doesn't do anything, to the program directly, instead, it helps us calculate the Delta of the Weights and the Bias.

            */

            layer.Delta.Data = (layer.NextLayer.WeightsT * layer.NextLayer.Delta).ElementMultiplication(layer.Values.Map(ActivationFuncDer)).Data;
        }
        public static void BackwardWeights(Layer layer, ref Matrix WeightDelta) {

            /*
            
            WD[L] += D[L] * A[L-1].T

            The change which we will apply to the Weights of that Layer.

            Note, we need the Weights to stay the same, otherwise the whole backpropagation is inaccurate.

            */

            WeightDelta += layer.Delta * layer.PreviousLayer.Activation.Transpose();
        }
        public static void BackwardBias(Layer layer, ref Vector BiasDelta) {
            
            /*
            
            BD[L] += D[L]

            The change which we will apply to the Bias of that Layer.

            Note, just like with the Weights, we must keep them unchanged during the backpropagation,
            else the results are incorrect.

            */
            
            BiasDelta += layer.Delta;
        }

        public static void AdjustWeight(Layer layer, ref Matrix WeightsDelta, float scale) {

            /*
            
            W[L] -= WD[L] * η / batchSize

            We have the Weight Delta, time to apply it.

            We have a new variable here called "Learning Rate" (η) which acts as the adjustment which each training example makes.
            If the Learning Rate was to be 1, then the network would train itself to perfectly give the correct output for that specific example.
            If it was 0, it wouldn't learn anything at all.

            */

            layer.Weights -= WeightsDelta * scale;
            layer.WeightsT = layer.Weights.Transpose();
        }
        public static void AdjustBias(Layer layer, ref Vector BiasDelta, float scale) {

            /*
            
            B[L] -= BD[L] * η / batchSize

            Just like with the Weights Delta, we apply the Bias Delta to the Bias.

            */

            layer.Bias -= BiasDelta * scale;
        }
    }
    public abstract class ParallelFunctions {
        public static void CalculateValue(Layer layer, Func<float, float> ActivationFunc, Vector input) {
            int simd_width = Vector<float>.Count;

            // Weights * Input + Bias

            for (int row = 0; row < layer.Weights.Rows; row++) {
                float sum = layer.Bias[row];
                int offset = row * layer.Weights.Columns;

                int col = 0;
                for (; col <= layer.Weights.Columns - simd_width; col += simd_width) {
                    var v_weights = new Vector<float>(layer.Weights.Data, offset + col);
                    var v_x = new Vector<float>(input.Data, col);
                    sum += System.Numerics.Vector.Dot(v_weights, v_x);
                }

                for (; col < layer.Weights.Columns; col++) {
                    sum += layer.Weights.Data[offset + col] * input.Data[col];
                }

                layer.Values[row] = sum;
                layer.Activation[row] = ActivationFunc(sum);
            }
        }

        public static void Backward(Layer layer, Func<float, float> ActivationFuncDer, Vector CorrectValues) { // input only used in output layer
            int simd_width = Vector<float>.Count; // 8

            // Weight is transposed, so rows and columns are reversed.

            int Rows = layer.NextLayer.WeightsT.Rows;
            int Columns = layer.NextLayer.WeightsT.Columns;

            //UnityEngine.Debug.Log($"NextLayer.Delta: {NextLayer.Delta.Length}, NextLayer.Weights: {NextLayer.Weights.Rows}x{NextLayer.Weights.Columns}, Delta: {Delta.Length}, NextLayer.Values: {NextLayer.Values.Length}");
            for (int row = 0; row < Rows; row++) {
                float sum = 0f;
                int offset = row * Columns;

                // Sum of N.W.T[row,...] * N.D[...]
                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v_weights = new Vector<float>(layer.NextLayer.WeightsT.Data, offset + col);
                    var v_delta = new Vector<float>(layer.NextLayer.Delta.Data, col);
                    sum += System.Numerics.Vector.Dot(v_weights, v_delta);
                }

                for (; col < Columns; col++) {
                    sum += layer.NextLayer.WeightsT.Data[offset + col] * layer.NextLayer.Delta.Data[col];
                }

                // Sum * ReLU'(Z[row])

                layer.Delta.Data[row] = sum * ActivationFuncDer(layer.Values.Data[row]);
            }
        }
        public static void BackwardWeights(Layer layer, ref Matrix WeightDelta) {
            int simd_width = Vector<float>.Count; // 8

            int Rows = layer.Delta.Length;
            int Columns = layer.PreviousLayer.Activation.Length;


            for (int row = 0; row < Rows; row++) {
                int offset = row * Columns;

                int col = 0;
                for (; col <= Columns - simd_width; col += simd_width) {
                    var v = new Vector<float>(layer.PreviousLayer.Activation.Data, col);
                    var v_weight = new Vector<float>(WeightDelta.Data, offset + col);

                    v = layer.Delta.Data[row] * v;
                    (v + v_weight).CopyTo(WeightDelta.Data, offset + col);
                }

                for (; col < Columns; col++) {
                    WeightDelta.Data[offset + col] += layer.Delta.Data[row] * layer.PreviousLayer.Activation.Data[col];
                }
            }
        }
        public static void BackwardBias(Layer layer, ref Vector BiasDelta) {
            int simd_width = Vector<float>.Count; // 8

            int Columns = layer.Delta.Length;

            int col = 0;
            for (; col <= Columns - simd_width; col += simd_width) {
                var v = new Vector<float>(layer.Delta.Data, col);
                var v_bias = new Vector<float>(BiasDelta.Data, col);
                (v + v_bias).CopyTo(BiasDelta.Data, col);
            }

            for (; col < Columns; col++) {
                BiasDelta.Data[col] += layer.Delta.Data[col];
            }
        }

        public static void AdjustWeight(Layer layer, ref Matrix WeightsDelta, float scale) {
            if (layer.Weights.Rows != WeightsDelta.Rows) throw new Exception("Weights and WeightsDelta don't have matching Rows!");
            if (layer.Weights.Columns != WeightsDelta.Columns) throw new Exception("Weights and WeightsDelta don't have matching Columns!");

            int simd_width = Vector<float>.Count;
            int length = layer.Weights.Rows * layer.Weights.Columns;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = new Vector<float>(WeightsDelta.Data, i);
                v_delta *= scale;
                var v = new Vector<float>(layer.Weights.Data, i);
                (v - v_delta).CopyTo(layer.Weights.Data, i);
            }
            for (; i < length; i++) {
                layer.Weights.Data[i] -= WeightsDelta.Data[i] * scale;
            }

            for (int row = 0; row < layer.WeightsT.Rows; row++) {
                for (int col = 0; col < layer.WeightsT.Columns; col++) {
                    layer.WeightsT[row, col] -= WeightsDelta[col, row] * scale;
                }
            }
        }
        public static void AdjustBias(Layer layer, ref Vector BiasDelta, float scale) {
            if (layer.Bias.Length != BiasDelta.Length) throw new Exception("Bias and BiasDelta don't have matching Lengths!");

            int simd_width = Vector<float>.Count;
            int length = layer.Bias.Length;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_delta = new Vector<float>(BiasDelta.Data, i);
                v_delta *= scale;
                var v = new Vector<float>(layer.Bias.Data, i);
                (v - v_delta).CopyTo(layer.Bias.Data, i);
            }
            for (; i < length; i++) {
                layer.Bias.Data[i] -= BiasDelta.Data[i] * scale;
            }
        }
    }
    public abstract class SSEFunctions {

    }
    public abstract class AVXFunctions {

    }
}
