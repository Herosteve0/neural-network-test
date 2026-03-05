using System;

namespace NeuralNetworkSystem {
    public class Layer {
        float WeightScaler(int previousLength) {
            return UnityEngine.Mathf.Sqrt(6f / previousLength);
        }

        public Layer(int size, LayerFunctions functions) {
            NeuronNum = size;
            Bias = new Vector(size);
            Values = new Vector(size);
            Activation = new Vector(size);
            Delta = new Vector(size);

            _Forward = functions.forward;
            _NormalizeInput = functions.normalization;
            _OutputActivation = functions.output_activation;

            _Backward= functions.backward_delta;
            _BackwardOutput = functions.backward_output_delta;
            _BackwardWeights = functions.backward_delta_weights;
            _BackwardBias = functions.backward_delta_bias;

            _AdjustWeights = functions.adjustment_weights;
            _AdjustBias = functions.adjustment_bias;

            ActivationFunction = functions.activation_function;
            ActivationFunctionDerivative = functions.activation_function_derivative;
        }
        public Layer(int size, Layer previousLayer, LayerFunctions functions) : this(size, functions) { // [to, from]
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


        protected ForwardPass _Forward;
        protected InputNormalization _NormalizeInput;
        protected OutputActivation _OutputActivation;

        protected BackwardDelta _Backward;
        protected BackwardOutputDelta _BackwardOutput;
        protected BackwardDeltaWeights _BackwardWeights;
        protected BackwardDeltaBias _BackwardBias;

        protected AdjustmentWeights _AdjustWeights;
        protected AdjustmentBias _AdjustBias;


        Func<float, float> ActivationFunction;
        Func<float, float> ActivationFunctionDerivative;


        public virtual void Forward(Vector input) { _Forward(this, ActivationFunction, input); }

        // input only used in output layer
        public virtual void Backward(Vector CorrectValues) { _Backward(this, ActivationFunctionDerivative); }
        public virtual void BackwardWeights(ref Matrix WeightDelta) { _BackwardWeights(this, ref WeightDelta); }
        public virtual void BackwardBias(ref Vector BiasDelta) { _BackwardBias(this, ref BiasDelta); }

        public virtual void AdjustWeight(ref Matrix WeightsDelta, float scale) { _AdjustWeights(this, ref WeightsDelta, scale); }
        public virtual void AdjustBias(ref Vector BiasDelta, float scale) { _AdjustBias(this, ref BiasDelta, scale); }
    }

    class InputLayer : Layer {
        public InputLayer(int size, LayerFunctions functions) : base(size, functions) { }

        public override void Forward(Vector input) {
            Activation.Data = _NormalizeInput(input.Data);
        }
    }

    class OutputLayer : Layer {
        public OutputLayer(int size, Layer previousLayer, LayerFunctions functions) : base(size, previousLayer, functions) { }

        public override void Forward(Vector input) {
            base.Forward(input);
            Activation.Data = _OutputActivation(Values.Data);
        }

        public override void Backward(Vector CorrectValues) { _BackwardOutput(this, CorrectValues); }
    }

    public class NeuralNetwork {
        public NeuralNetwork(int[] layers, LayerFunctions functions) {
            LayerAmount = layers.Length;
            Layers = new Layer[LayerAmount];
            LayerLength = new int[LayerAmount];

            for (int i = 0; i < LayerAmount; i++) {
                LayerLength[i] = layers[i];
                if (i == 0) {
                    Layers[0] = new InputLayer(layers[i], functions);
                } else if (i == LayerAmount - 1) {
                    Layers[LayerAmount - 1] = new OutputLayer(layers[i], Layers[i - 1], functions);
                } else {
                    Layers[i] = new Layer(layers[i], Layers[i - 1], functions);
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