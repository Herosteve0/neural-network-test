using System;
using System.Numerics;
using Unity.VisualScripting.Antlr3.Runtime;
//using System.Runtime.Instricts;
//using System.Runtime.Instricts.x86;

namespace NeuralNetworkSystem {
    public delegate float[] InputNormalization(float[] input);
    public delegate float[] OutputActivation(float[] input);
    public delegate float LossCalculation(float[] V, int label);

    public delegate void ForwardPass(Layer layer, Func<float, float> ActivationFunc, Vector input);

    public delegate void BackwardDelta(Layer layer, Func<float, float> ActivationFuncDer);
    public delegate void BackwardOutputDelta(Layer layer, Vector CorrectValues);
    public delegate void BackwardDeltaWeights(Layer layer, ref Matrix WeightDelta);
    public delegate void BackwardDeltaBias(Layer layer, ref Vector BiasDelta);

    public delegate void AdjustmentWeights(Layer layer, ref Matrix WeightsDelta, float scale);
    public delegate void AdjustmentBias(Layer layer, ref Vector BiasDelta, float scale);

    public struct LayerFunctions {
        public ForwardPass forward;
        public InputNormalization normalization;
        public OutputActivation output_activation;

        public BackwardDelta backward_delta;
        public BackwardOutputDelta backward_output_delta;
        public BackwardDeltaWeights backward_delta_weights;
        public BackwardDeltaBias backward_delta_bias;

        public AdjustmentWeights adjustment_weights;
        public AdjustmentBias adjustment_bias;

        public Func<float, float> activation_function;
        public Func<float, float> activation_function_derivative;

        public LayerFunctions(
            ForwardPass forward,
            InputNormalization normalization,
            OutputActivation output_activation,

            BackwardDelta backward_delta,
            BackwardOutputDelta backward_output_delta,
            BackwardDeltaWeights backward_delta_weights,
            BackwardDeltaBias backward_delta_bias,

            AdjustmentWeights adjustment_weights,
            AdjustmentBias adjustment_bias,

            Func<float, float> activation_function,
            Func<float, float> activation_function_derivative
            ) {

            this.forward = forward;
            this.normalization = normalization;
            this.output_activation = output_activation;

            this.backward_delta = backward_delta;
            this.backward_output_delta = backward_output_delta;
            this.backward_delta_weights = backward_delta_weights;
            this.backward_delta_bias = backward_delta_bias;

            this.adjustment_weights = adjustment_weights;
            this.adjustment_bias = adjustment_bias;

            this.activation_function = activation_function;
            this.activation_function_derivative = activation_function_derivative;
        }
    }

    public class FunctionManager {
        public static Func<float, float> GetActivationFunction(ActivationFunctionsTypes type) {
            switch (type) {
                case ActivationFunctionsTypes.Sigmoid: return ActivationFunctions.Sigmoid;
                case ActivationFunctionsTypes.ReLU: return ActivationFunctions.ReLU;
            }
            return null;
        }
        public static Func<float, float> GetActivationDerivativeFunction(ActivationFunctionsTypes type) {
            switch (type) {
                case ActivationFunctionsTypes.Sigmoid: return ActivationFunctions.SigmoidDerivative;
                case ActivationFunctionsTypes.ReLU: return ActivationFunctions.ReLUDerivative;
            }
            return null;
        }
        public static InputNormalization GetInputNormalizationFunction(InputNormalizationFunctionsType type) {
            switch (type) {
                case InputNormalizationFunctionsType.Normalize: return InputNormalizationFunctions.Normalize;
                case InputNormalizationFunctionsType.NormalizeMeadian: return InputNormalizationFunctions.NormalizeMedian;
            }
            return null;
        }
        public static OutputActivation GetOutputFunction(OutputFunctionsType type) {
            switch (type) {
                case OutputFunctionsType.SoftMax: return OutputFunctions.SoftMax;
            }
            return null;
        }
        public static LossCalculation GetLossFunction(LossFunctionsType type) {
            switch (type) {
                case LossFunctionsType.Mean: return LossFunctions.Mean;
                case LossFunctionsType.SoftMax: return LossFunctions.SoftMax;
            }
            return null;
        }


        public static ForwardPass GetForwardFunction(ForwardFunctionsTypes type) {
            switch (type) {
                case ForwardFunctionsTypes.Educational: return EducationalFunctions.CalculateValue;
                case ForwardFunctionsTypes.Scalar: return ScalarFunctions.CalculateValue;
                case ForwardFunctionsTypes.Parallel: return ParallelFunctions.CalculateValue;
                case ForwardFunctionsTypes.SSE: return SSEFunctions.CalculateValue;
                case ForwardFunctionsTypes.AVX: return AVXFunctions.CalculateValue;
            }
            return null;
        }

        public static BackwardDelta GetBackwardFunction(BackwardFunctionsTypes type) {
            switch (type) {
                case BackwardFunctionsTypes.Educational: return EducationalFunctions.Backward;
                case BackwardFunctionsTypes.Scalar: return ScalarFunctions.Backward;
                case BackwardFunctionsTypes.Parallel: return ParallelFunctions.Backward;
                case BackwardFunctionsTypes.SSE: return SSEFunctions.Backward;
                case BackwardFunctionsTypes.AVX: return AVXFunctions.Backward;
            }
            return null;
        }
        public static BackwardOutputDelta GetBackwardOutputFunction(BackwardOutputFunctionsTypes type) {
            switch (type) {
                case BackwardOutputFunctionsTypes.Educational: return EducationalFunctions.BackwardOutput;
                case BackwardOutputFunctionsTypes.Scalar: return ScalarFunctions.BackwardOutput;
                case BackwardOutputFunctionsTypes.Parallel: return ParallelFunctions.BackwardOutput;
                case BackwardOutputFunctionsTypes.SSE: return SSEFunctions.BackwardOutput;
                case BackwardOutputFunctionsTypes.AVX: return AVXFunctions.BackwardOutput;
            }
            return null;
        }
        public static BackwardDeltaWeights GetBackwardWeightsFunction(BackwardWeightsFunctionsTypes type) {
            switch (type) {
                case BackwardWeightsFunctionsTypes.Educational: return EducationalFunctions.BackwardWeights;
                case BackwardWeightsFunctionsTypes.Scalar: return ScalarFunctions.BackwardWeights;
                case BackwardWeightsFunctionsTypes.Parallel: return ParallelFunctions.BackwardWeights;
                case BackwardWeightsFunctionsTypes.SSE: return SSEFunctions.BackwardWeights;
                case BackwardWeightsFunctionsTypes.AVX: return AVXFunctions.BackwardWeights;
            }
            return null;
        }
        public static BackwardDeltaBias GetBackwardBiasFunction(BackwardBiasFunctionsTypes type) {
            switch (type) {
                case BackwardBiasFunctionsTypes.Educational: return EducationalFunctions.BackwardBias;
                case BackwardBiasFunctionsTypes.Scalar: return ScalarFunctions.BackwardBias;
                case BackwardBiasFunctionsTypes.Parallel: return ParallelFunctions.BackwardBias;
                case BackwardBiasFunctionsTypes.SSE: return SSEFunctions.BackwardBias;
                case BackwardBiasFunctionsTypes.AVX: return AVXFunctions.BackwardBias;
            }
            return null;
        }

        public static AdjustmentWeights GetAdjustWeightsFunction(AdjustWeightsFunctionsTypes type) {
            switch (type) {
                case AdjustWeightsFunctionsTypes.Educational: return EducationalFunctions.AdjustWeights;
                case AdjustWeightsFunctionsTypes.Scalar: return ScalarFunctions.AdjustWeights;
                case AdjustWeightsFunctionsTypes.Parallel: return ParallelFunctions.AdjustWeights;
                case AdjustWeightsFunctionsTypes.SSE: return SSEFunctions.AdjustWeights;
                case AdjustWeightsFunctionsTypes.AVX: return AVXFunctions.AdjustWeights;
            }
            return null;
        }
        public static AdjustmentBias GetAdjustBiasFunction(AdjustBiasFunctionsTypes type) {
            switch (type) {
                case AdjustBiasFunctionsTypes.Educational: return EducationalFunctions.AdjustBias;
                case AdjustBiasFunctionsTypes.Scalar: return ScalarFunctions.AdjustBias;
                case AdjustBiasFunctionsTypes.Parallel: return ParallelFunctions.AdjustBias;
                case AdjustBiasFunctionsTypes.SSE: return SSEFunctions.AdjustBias;
                case AdjustBiasFunctionsTypes.AVX: return AVXFunctions.AdjustBias;
            }
            return null;
        }


        public static LayerFunctions GetFunctions(
                ForwardFunctionsTypes forward,
                InputNormalizationFunctionsType normalization,
                OutputFunctionsType output_activation,

                BackwardFunctionsTypes backward_delta,
                BackwardOutputFunctionsTypes backward_output_delta,
                BackwardWeightsFunctionsTypes backward_delta_weights,
                BackwardBiasFunctionsTypes backward_delta_bias,

                AdjustWeightsFunctionsTypes adjustment_weights,
                AdjustBiasFunctionsTypes adjustment_bias,

                ActivationFunctionsTypes activation_function
            ) {
            return new LayerFunctions(
                GetForwardFunction(forward),
                GetInputNormalizationFunction(normalization),
                GetOutputFunction(output_activation),

                GetBackwardFunction(backward_delta),
                GetBackwardOutputFunction(backward_output_delta),
                GetBackwardWeightsFunction(backward_delta_weights),
                GetBackwardBiasFunction(backward_delta_bias),

                GetAdjustWeightsFunction(adjustment_weights),
                GetAdjustBiasFunction(adjustment_bias),

                GetActivationFunction(activation_function),
                GetActivationDerivativeFunction(activation_function)
                );
        }
    }

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

    public enum InputNormalizationFunctionsType {
        Normalize,
        NormalizeMeadian
    }
    public abstract class InputNormalizationFunctions {
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

        public static float SoftMax(float[] V, int label) {
            return -UnityEngine.Mathf.Log(V[label]);
        }
    }

    public enum ForwardFunctionsTypes {
        Educational,
        Scalar,
        Parallel,
        SSE,
        AVX
    }
    public enum BackwardFunctionsTypes {
        Educational,
        Scalar,
        Parallel,
        SSE,
        AVX
    }
    public enum BackwardOutputFunctionsTypes {
        Educational,
        Scalar,
        Parallel,
        SSE,
        AVX
    }
    public enum BackwardWeightsFunctionsTypes {
        Educational,
        Scalar,
        Parallel,
        SSE,
        AVX
    }
    public enum BackwardBiasFunctionsTypes {
        Educational,
        Scalar,
        Parallel,
        SSE,
        AVX
    }
    public enum AdjustWeightsFunctionsTypes {
        Educational,
        Scalar,
        Parallel,
        SSE,
        AVX
    }
    public enum AdjustBiasFunctionsTypes {
        Educational,
        Scalar,
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
        public static void BackwardOutput(Layer layer, Vector CorrectValues) {

            /*
            
            D[L] = A[L] - Y

            Since the last layer does not have a layer to be compared to and we need to start the chain somehow, we use the correct answer (Y)
            as a reference to say "This answer in a perfect world looks like this". This value is directly tied to the Loss.
            Realistically, the Loss will never be a true 0, since that'd mean it only recognizes that answer, but the lowest it can get, the better.

            */

            layer.Delta.Data = (layer.Activation - CorrectValues).Data;
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

        public static void AdjustWeights(Layer layer, ref Matrix WeightsDelta, float scale) {

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
    public abstract class ScalarFunctions {
        public static void CalculateValue(Layer layer, Func<float, float> ActivationFunc, Vector input) {
            int Rows = layer.Values.Length;
            int Columns = layer.Weights.Columns;

            for (int row = 0; row < Rows; row++) {
                layer.Values[row] = layer.Bias[row];

                for (int col = 0; col < Columns; col++) {
                    layer.Values[row] += layer.Weights[row, col] * input[col];
                }

                layer.Activation[row] = ActivationFunc(layer.Values[row]);
            }
        }

        public static void Backward(Layer layer, Func<float, float> ActivationFuncDer) {
            int Rows = layer.Delta.Length;
            int Columns = layer.WeightsT.Columns;

            for (int row = 0; row < Rows; row++) {
                layer.Delta[row] = 0f;

                for (int col = 0; col < Columns; col++) {
                    layer.Delta[row] += layer.NextLayer.WeightsT[row, col] * layer.NextLayer.Delta[col];
                }

                layer.Delta[row] *= ActivationFuncDer(layer.Values[row]);
            }
        }
        public static void BackwardOutput(Layer layer, Vector CorrectValues) {
            int Rows = layer.Activation.Length;

            for (int row = 0; row < Rows; row++) {
                layer.Delta[row] = layer.Activation[row] - CorrectValues[row];
            }
        }
        public static void BackwardWeights(Layer layer, ref Matrix WeightDelta) {
            int Rows = layer.Delta.Length;
            int Columns = layer.Activation.Length;

            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Columns; col++) {
                    WeightDelta[row, col] += layer.Delta[row] * layer.Activation[col];
                }
            }
        }
        public static void BackwardBias(Layer layer, ref Vector BiasDelta) {
            int Rows = layer.Delta.Length;

            for (int row = 0; row < Rows; row++) {
                BiasDelta[row] += layer.Delta[row];
            }
        }

        public static void AdjustWeights(Layer layer, ref Matrix WeightsDelta, float scale) {
            int Rows = WeightsDelta.Rows;
            int Columns = WeightsDelta.Columns;

            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Columns; col++) {
                    layer.Weights[row, col] -= WeightsDelta[row, col] * scale;
                    layer.WeightsT[col, row] -= WeightsDelta[col, row] * scale;
                }
            }
        }
        public static void AdjustBias(Layer layer, ref Vector BiasDelta, float scale) {
            int Rows = layer.Bias.Length;

            for (int row = 0; row < Rows; row++) {
                layer.Bias[row] -= BiasDelta[row];
            }
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

        public static void Backward(Layer layer, Func<float, float> ActivationFuncDer) { // input only used in output layer
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
        public static void BackwardOutput(Layer layer, Vector CorrectValues) {
            int simd_width = Vector<float>.Count;

            int Rows = layer.Activation.Length;

            int i = 0;
            for (; i <= Rows - simd_width; i += simd_width) {
                var v_a = new Vector<float>(layer.Activation.Data, i);
                var v_b = new Vector<float>(CorrectValues.Data, i);
                (v_a - v_b).CopyTo(layer.Delta.Data, i);
            }
            for (; i < Rows; i++) {
                layer.Delta.Data[i] = layer.Activation[i] - CorrectValues[i];
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

        public static void AdjustWeights(Layer layer, ref Matrix WeightsDelta, float scale) {
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
        public static void CalculateValue(Layer layer, Func<float, float> ActivationFunc, Vector input) { }

        public static void Backward(Layer layer, Func<float, float> ActivationFuncDer) { }
        public static void BackwardOutput(Layer layer, Vector CorrectValues) { }
        public static void BackwardWeights(Layer layer, ref Matrix WeightDelta) { }
        public static void BackwardBias(Layer layer, ref Vector BiasDelta) { }

        public static void AdjustWeights(Layer layer, ref Matrix WeightsDelta, float scale) { }
        public static void AdjustBias(Layer layer, ref Vector BiasDelta, float scale) { }
    }
    public abstract class AVXFunctions {
        public static void CalculateValue(Layer layer, Func<float, float> ActivationFunc, Vector input) { }

        public static void Backward(Layer layer, Func<float, float> ActivationFuncDer) { }
        public static void BackwardOutput(Layer layer, Vector CorrectValues) { }
        public static void BackwardWeights(Layer layer, ref Matrix WeightDelta) { }
        public static void BackwardBias(Layer layer, ref Vector BiasDelta) { }

        public static void AdjustWeights(Layer layer, ref Matrix WeightsDelta, float scale) { }
        public static void AdjustBias(Layer layer, ref Vector BiasDelta, float scale) { }

    }
}
