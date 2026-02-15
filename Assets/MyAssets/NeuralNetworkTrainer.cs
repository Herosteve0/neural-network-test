using System;
using System.Text;
using UnityEngine.UIElements;

namespace NeuralNetworkSystem {
    public struct Data {
        public Data(Vector data, int label) {
            this.data = data;
            this.label = label;
        }

        public Vector data { get; }
        public int label { get; }
    }

    public class NeuralNetworkTrainer {
        public NeuralNetworkTrainer(NeuralNetwork network, Data[] data, float learning_rate) {
            Network = network;
            TrainingData = data;
            this.learning_rate = learning_rate;
        }

        NeuralNetwork Network { get; }
        public Data[] TrainingData { get; }
        public float learning_rate { get; }


        public static float Sigmoid(float value) {
            float k = (float)Math.Exp(-value);
            return 1 / (1 + k);
        }

        // f(x) * f(-x)
        // f(x) * (1 - f(x))
        // e^(-x) / (1 + e^(-x))^2
        public static float SigmoidDerivative(float value) {
            float k = (float)Math.Exp(-value);
            return k / ((1 + k) * (1 + k));
        }

        public static Vector Loss(Vector V, Vector Y) {
            Vector Cost = V - Y;
            return Cost.DotProduct(Cost);
        }

        public static Vector LossDerivative(Vector V, Vector Y) {
            return (V - Y) * 2;
        }

        public void SingleExampleTraining(int index) {
            if (index < 0 || index > TrainingData.Length) return;
            Network.Calculate(TrainingData[index].data); // all layers of the network have the values we want (inupt, value, activation)

            Vector Y = Vector.SingleValue(Network.LayerLength[Network.LayerAmount - 1], TrainingData[index].label);
            Vector Cost = Loss(Network.Layers[Network.LayerAmount - 1].Activation, Y);

            Vector[] Delta = new Vector[Network.LayerAmount - 1];
            Matrix[] WeightDelta = new Matrix[Network.LayerAmount - 1];
            Vector[] BiasDelta = new Vector[Network.LayerAmount - 1];

            Delta[Network.LayerAmount - 1] = LossDerivative(
                Network.Layers[Network.LayerAmount - 1].Activation, Y
                ).DotProduct(
                Network.Layers[Network.LayerAmount - 1].Values.Map(SigmoidDerivative)
                );

            for (int i = Network.LayerAmount - 1; i > 0; i--) {
                Delta[i - 1] = (Network.Layers[i].Weights.Transpose() * Delta[i]).DotProduct(Network.Layers[i - 1].Values.Map(SigmoidDerivative));

                WeightDelta[i] = Delta[i] * Network.Layers[i - 1].Activation.Transpose();
                BiasDelta[i] = Delta[i];
            }

            for (int i = 1; i < Network.LayerAmount; i++) {
                Network.Layers[i].Weights -= WeightDelta[i] * learning_rate;
                Network.Layers[i].Bias -= BiasDelta[i] * learning_rate;
            }
        }

        public void FullTraining() {
            for (int i = 0; i < TrainingData.Length; i++) {
                SingleExampleTraining(i);
            }
        }
    }
}