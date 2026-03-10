using System;
using System.Numerics;
//using System.Runtime.Instricts;
//using System.Runtime.Instricts.x86;

namespace NeuralNetworkSystem {
    /*
    
    These variables here are the functions the whole program needs in order to work.

    InputNormalization: A change we can do to our input in order to make it more suitable for our network.

    OutputActivation: The function which we will use in order to properly activate the neurons of the last layer, in order to get the correct values.
                      Note that this function is specifically for the output, since if we were to put it in the hidden layers, it would break the strength
                      of each activation.

    LossCalculation
    */
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

    // Warning, this code is NOT good, I just did it cause it was the simplest way I could think of doing it.
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
                case InputNormalizationFunctionsType.None: return InputNormalizationFunctions.None;
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
                case ForwardFunctionsTypes.SIMD: return SIMDFunctions.CalculateValue;
                //case ForwardFunctionsTypes.SSE: return SSEFunctions.CalculateValue;
                //case ForwardFunctionsTypes.AVX: return AVXFunctions.CalculateValue;
            }
            return null;
        }

        public static BackwardDelta GetBackwardFunction(BackwardFunctionsTypes type) {
            switch (type) {
                case BackwardFunctionsTypes.Educational: return EducationalFunctions.Backward;
                case BackwardFunctionsTypes.Scalar: return ScalarFunctions.Backward;
                case BackwardFunctionsTypes.SIMD: return SIMDFunctions.Backward;
                //case BackwardFunctionsTypes.SSE: return SSEFunctions.Backward;
                //case BackwardFunctionsTypes.AVX: return AVXFunctions.Backward;
            }
            return null;
        }
        public static BackwardOutputDelta GetBackwardOutputFunction(BackwardOutputFunctionsTypes type) {
            switch (type) {
                case BackwardOutputFunctionsTypes.Educational: return EducationalFunctions.BackwardOutput;
                case BackwardOutputFunctionsTypes.Scalar: return ScalarFunctions.BackwardOutput;
                case BackwardOutputFunctionsTypes.SIMD: return SIMDFunctions.BackwardOutput;
                //case BackwardOutputFunctionsTypes.SSE: return SSEFunctions.BackwardOutput;
                //case BackwardOutputFunctionsTypes.AVX: return AVXFunctions.BackwardOutput;
            }
            return null;
        }
        public static BackwardDeltaWeights GetBackwardWeightsFunction(BackwardWeightsFunctionsTypes type) {
            switch (type) {
                case BackwardWeightsFunctionsTypes.Educational: return EducationalFunctions.BackwardWeights;
                case BackwardWeightsFunctionsTypes.Scalar: return ScalarFunctions.BackwardWeights;
                case BackwardWeightsFunctionsTypes.SIMD: return SIMDFunctions.BackwardWeights;
                //case BackwardWeightsFunctionsTypes.SSE: return SSEFunctions.BackwardWeights;
                //case BackwardWeightsFunctionsTypes.AVX: return AVXFunctions.BackwardWeights;
            }
            return null;
        }
        public static BackwardDeltaBias GetBackwardBiasFunction(BackwardBiasFunctionsTypes type) {
            switch (type) {
                case BackwardBiasFunctionsTypes.Educational: return EducationalFunctions.BackwardBias;
                case BackwardBiasFunctionsTypes.Scalar: return ScalarFunctions.BackwardBias;
                case BackwardBiasFunctionsTypes.SIMD: return SIMDFunctions.BackwardBias;
                //case BackwardBiasFunctionsTypes.SSE: return SSEFunctions.BackwardBias;
                //case BackwardBiasFunctionsTypes.AVX: return AVXFunctions.BackwardBias;
            }
            return null;
        }

        public static AdjustmentWeights GetAdjustWeightsFunction(AdjustWeightsFunctionsTypes type) {
            switch (type) {
                case AdjustWeightsFunctionsTypes.Educational: return EducationalFunctions.AdjustWeights;
                case AdjustWeightsFunctionsTypes.Scalar: return ScalarFunctions.AdjustWeights;
                case AdjustWeightsFunctionsTypes.SIMD: return SIMDFunctions.AdjustWeights;
                //case AdjustWeightsFunctionsTypes.SSE: return SSEFunctions.AdjustWeights;
                //case AdjustWeightsFunctionsTypes.AVX: return AVXFunctions.AdjustWeights;
            }
            return null;
        }
        public static AdjustmentBias GetAdjustBiasFunction(AdjustBiasFunctionsTypes type) {
            switch (type) {
                case AdjustBiasFunctionsTypes.Educational: return EducationalFunctions.AdjustBias;
                case AdjustBiasFunctionsTypes.Scalar: return ScalarFunctions.AdjustBias;
                case AdjustBiasFunctionsTypes.SIMD: return SIMDFunctions.AdjustBias;
                //case AdjustBiasFunctionsTypes.SSE: return SSEFunctions.AdjustBias;
                //case AdjustBiasFunctionsTypes.AVX: return AVXFunctions.AdjustBias;
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
        /*
         
        f(x) = 1 / (e^(-x) + 1), f: R -> (0, 1)
        f'(x) = e^(-x) / ( (e^(-x) + 1)^2 ) = f(x) * ( 1 - f(x) )
        
        This function confies all numbers into the range (0, 1),
        it returns a number close to 0 the closer the number is to negative infinity and close to 1 the closer the number is to positive infinity

        Sigmoid heavily punishes bad Neurons, while heavily rewarding good Neurons.

        */

        public static float Sigmoid(float value) {
            float e = (float)Math.Exp(-value);
            return 1 / (e + 1);
        }
        public static float SigmoidDerivative(float value) {
            float a = Sigmoid(value);
            return a * (1 - a);
        }

        /*
        
        f(x) = { x, x > 0       , f: R -> [0, + infinity)
               { 0, x <= 0

        This function stops any negative value from proceeding.

        ReLU keeps positive signals while supressing negative ones. In a more general sense, it only uses anything it can take advantage of and ignores anything it doesn't find worthy.
        This function is especially important for Transformers (LLM, GPT)
        
        */

        public static float ReLU(float value) {
            return value >= 0 ? value : 0;
        }
        public static float ReLUDerivative(float value) {
            return value > 0 ? 1 : 0;
        }
    }

    public enum InputNormalizationFunctionsType {
        None,
        NormalizeMeadian
    }
    public abstract class InputNormalizationFunctions {


        /*
        
        For this project, we use the MNIST database, which gives us the gray scale values of images and the handler for that transforms that into floats from [0,1], with 0 being the value 0 and 255 being the value 1.
        None is pretty much "What if we used these numbers directly?" which isn't a bad approach, however it might take a bit more time for the network to adjust to using the range [0,1]

        */

        public static float[] None(float[] input) {
            return input;
        }

        /*
        
        NormalizeMedian essentially helps the network by slightly adjust the inputs for it. Instead of having a value [0, 1], we now instead have a value that relates to the pixel value compared to all other pixels.
        This our database is a solved problem, we can chuck the values directly (look at "mean" and "std" variables), however if you wanted to calculate the values yourself, you'd do:

        mean = sum      

        */

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

        SoftMax is a function that takes many values and returns the probability distribution of these values.
        The sum of this Vector will always be 1.

        */
        public static float[] SoftMax(float[] output) {
            int length = output.Length;
            float[] r = new float[length];

            float max = output[0];
            for (int i = 1; i < length; i++) {
                if (max < output[i]) max = output[i];
            }

            float sum = 0f;
            for (int i = 0; i < length; i++) {
                float e = (float)Math.Exp(output[i] - max);
                r[i] = e;
                sum += e;
            }

            for (int i = 0; i < length; i++) {
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
            float a = V[label] - 1f;
            return a * a;
        }

        public static float SoftMax(float[] V, int label) {
            return -UnityEngine.Mathf.Log(V[label]);
        }
    }

    public enum ForwardFunctionsTypes {
        Educational,
        Scalar,
        SIMD,
        //SSE,
        //AVX
    }
    public enum BackwardFunctionsTypes {
        Educational,
        Scalar,
        SIMD,
        //SSE,
        //AVX
    }
    public enum BackwardOutputFunctionsTypes {
        Educational,
        Scalar,
        SIMD,
        //SSE,
        //AVX
    }
    public enum BackwardWeightsFunctionsTypes {
        Educational,
        Scalar,
        SIMD,
        //SSE,
        //AVX
    }
    public enum BackwardBiasFunctionsTypes {
        Educational,
        Scalar,
        SIMD,
        //SSE,
        //AVX
    }
    public enum AdjustWeightsFunctionsTypes {
        Educational,
        Scalar,
        SIMD,
        //SSE,
        //AVX
    }
    public enum AdjustBiasFunctionsTypes {
        Educational,
        Scalar,
        SIMD,
        //SSE,
        //AVX
    }

    /*
    
    The Educational Functions are for people who want to take a first look at the math of Neural Networks.
    These functions are Scalar, meaning that each calculation is done one after the other with no parallel calculations.

    While the code looks okay and clean, the main problem with this approach is ram allocation.
    Inside "MathStructs.cs" you can see the code each operator and the first thing you'll see is how it returns a new Vector/Matrix each time.
    
    For example, V = Wx+B

    For simplicity, we will call N[L] the size of the current layer L and the size of the previous layer N[L-1]

    For starters, we have the variables:

    V: N[L]
    W: N[L] * N[L-1]
    x: N[L-1]
    B: N[L]

            V          W             x      B
    Total: N[L] + N[L] * N[L-1] + N[L-1] + N[L] = 2N[L] + N[L-1] + N[L] * N[L-1]

    When the calculation W*x happens, we create a new Vector with size N[L] and we will call this Vector tmp
    Then, we have the calculation tmp + B, which creates a new Vector of size N[L], we will call this one R
    Lastly, we set the data of the Vector V to the data of the Vector R.

                                                tmp     R
    So now, our total suddenly went to: Total + N[L] + N[L] = 4N[L] + N[L-1] + N[L] * N[L-1]
    The percentage of how much extra memory this takes is around 2N[L] / Total, however in bigger models where memory is finite, we need to do better.

    Small note that it is still possible to do implement this without the memory issues, however for reason you'll see in Scalar, it's not worth it.
    For those who are curious though, you can simply make each operator a function of the class. So instead of W*x. you'd have W.Multiply(x).

    */
    public abstract class EducationalFunctions {
        public static void CalculateValue(Layer layer, Func<float, float> ActivationFunc, Vector input) {

            /*
            
            Forward is essentially the prediction function, in a broad sense, the Network tries to predict the what the output should be using the input it got.

            
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
            
            Backward looks at what it got, it looks at what it was supposed to get and thinks of a way to become better.
            
            
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


    /*
    
    Scalar has the first big improvement we can make in our algorithm, which is both in regards to memory (look at Educational) and in time.
    The Educational Functions did each computation one by one, meaning we first calculated W*x and then added B, when we can simply do in parallel.

    Once again, the amount of Neurons in a Layer is N[L]
    
    Our mental model now moves towards the notation
    
    -------------------------------W*x+B------------------------------
                   --------------------W[i]*x+B[i]--------------------
                            ------------------W[i]*x------------------
    Σ{i: 1->N[L] } ( B[i] + Σ{j: 1 -> N[L-1] } ( W[i, j] * x[j] ) ) 

    So now, not only do we avoid the additional memory Educational Functions created,
    but we also save time, since we looping N[L] * N[L-1] times, instead of the N[L] * N[L-1] + N[L] ( W*x is O(N[L]*N[L-1]) and +B is O(N[L]) )

    */
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
            int Columns = layer.NextLayer.WeightsT.Columns;

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
            int Columns = layer.PreviousLayer.Activation.Length;

            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Columns; col++) {
                    WeightDelta[row, col] += layer.Delta[row] * layer.PreviousLayer.Activation[col];
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
                    layer.WeightsT[col, row] -= WeightsDelta[row, col] * scale;
                }
            }
        }
        public static void AdjustBias(Layer layer, ref Vector BiasDelta, float scale) {
            int Rows = BiasDelta.Length;

            for (int row = 0; row < Rows; row++) {
                layer.Bias[row] -= BiasDelta[row] * scale;
            }
        }
    }


    /*
    
    SIMD (Single Instruction, Multiple Data) is the CPU's way of doing multiple instructions at once. You take multiple data and execute the same idea on all of them at once.
    For Intel and AMD CPUs, there is SSE (128 bits), AVX (256 bits) and AVX-512 (512 bits) (SSE and AVX are not supported in Unity, cause of .NET 2.1).
    For Arm CPUs, it has Neon which uses 64 or 128 bits.

    .NET 2.1 has Vector<T>, which is what the following functions use.

    Example:
        Let's say we have 4 floats a0, b0, c0 and d0 and we want to multiple them with a1, b1, c1 and d1 respectively.
        
        One logical way to do so is one step at a time (scalar)
          a0 * a1
          b0 * b1
          c0 * c1
          d0 * d1
        each multiplication requires one instruction.
    
        What SIMD would do is instead is group the numbers in a vector and process them as
        [a0, b0, c0, d0] * [a1, b1, c1, d1]
                         ↓
        [a0 * a1, b0 * b1, c0 * c1, d0 * d1]
        So now, instead of 4 separate instructions, we do them all at once.

    The reason I used four numbers, is because in this version we also use 4.
    .NET 2.1 used SSE, which has 128 bits, and our program uses floats (32 bits) so we can use Vectors with length 128 / 32 = 4 (shown by Vector<float>.Count)
    Another note, due to the format of the Vector<T> initialization, if the elements we are about to put in the Vector are less then 4, we are best off using normal Scalar, as you will see in the code.

    */
    public abstract class SIMDFunctions {
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
            int simd_width = Vector<float>.Count;

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
            int simd_width = Vector<float>.Count;

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
            int simd_width = Vector<float>.Count;

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
}
