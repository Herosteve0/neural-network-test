using System;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
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

    public class DataBatch {
        public DataBatch(Data[] data) {
            Data = data;
            Size = data.Length;
        }

        public Data[] Data { get; }
        public int Size { get; }
    }

    public class NeuralNetworkTrainer {
        public NeuralNetworkTrainer(NeuralNetwork network, float learning_rate, Func<int, Data> training_function, int training_amount) {
            Network = network;
            TrainingFunction = training_function;
            TrainingExamplesCount = training_amount;
            this.learning_rate = learning_rate;
        }
        public NeuralNetworkTrainer(NeuralNetwork network, float learning_rate) {
            Network = network;
            this.learning_rate = learning_rate;
        }

        NeuralNetwork Network { get; }
        public Func<int, Data> TrainingFunction { get; }
        public int TrainingExamplesCount { get; }
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

        public static float SoftMaxLoss(Vector V, int label) {
            return -UnityEngine.Mathf.Log(V[label]);
        }
        public static float Loss(Vector V, int label) {
            float Cost = V[label] - 1;
            return Cost * Cost;
        }

        public static float LossDerivative(Vector V, int label) {
            return (V[label] - 1) * 2;
        }

        public void TrainingCalculations(Data TrainingData, ref Vector[] Delta, ref Matrix[] WeightDelta, ref Vector[] BiasDelta) {
            Network.Calculate(TrainingData.data); // all layers of the network have the values we want (inupt, value, activation)

            int length = Network.LayerAmount - 1;

            Vector Y = Vector.SingleValue(Network.LayerLength[length], TrainingData.label);
            float Loss = SoftMaxLoss(Network.Layers[length].Activation, TrainingData.label);
            Vector Cost = Vector.SingleValue(Network.LayerLength[length] , TrainingData.label, Loss);

            for (int i = length - 1; i >= 0; i--) {
                int l = i + 1;
                if (i == length - 1) {
                    //Delta[i] = LossDerivative(
                    //    Network.Layers[Network.LayerAmount - 1].Activation, Y
                    //    ).DotProduct(
                    //    Network.Layers[Network.LayerAmount - 1].Values.Map(SigmoidDerivative)
                    //    );
                    Delta[i] = Network.Layers[l].Activation - Y;
                } else {
                    Delta[i] = (Network.Layers[l + 1].Weights.Transpose() * Delta[l]).DotProduct(Network.Layers[l].Values.Map(SigmoidDerivative));
                }

                WeightDelta[i] += Delta[i] * Network.Layers[i].Activation.Transpose();
                BiasDelta[i] += Delta[i];
            }
        }

        public void SingleExampleTraining(Data TrainingData) {
            Vector[] Delta = new Vector[Network.LayerAmount - 1];
            Matrix[] WeightDelta = new Matrix[Network.LayerAmount - 1];
            Vector[] BiasDelta = new Vector[Network.LayerAmount - 1];

            for (int i = 0; i < Network.LayerAmount - 1; i++) {
                WeightDelta[i] = new Matrix(Network.LayerLength[i + 1], Network.LayerLength[i]);
                BiasDelta[i] = new Vector(Network.LayerLength[i + 1]);
            }

            TrainingCalculations(TrainingData, ref Delta, ref WeightDelta, ref BiasDelta);

            for (int i = 1; i < Network.LayerAmount; i++) {
                Network.Layers[i].Weights -= WeightDelta[i - 1] * learning_rate;
                Network.Layers[i].Bias -= BiasDelta[i - 1] * learning_rate;
            }
        }

        public void BatchTraining(DataBatch DataBatch) {
            Vector[] Delta = new Vector[Network.LayerAmount - 1];
            Matrix[] WeightDelta = new Matrix[Network.LayerAmount - 1];
            Vector[] BiasDelta = new Vector[Network.LayerAmount - 1];

            for (int i = 0; i < Network.LayerAmount - 1; i++) {
                WeightDelta[i] = new Matrix(Network.LayerLength[i + 1], Network.LayerLength[i]);
                BiasDelta[i] = new Vector(Network.LayerLength[i + 1]);
            }

            foreach (Data TrainingData in DataBatch.Data) {
                TrainingCalculations(TrainingData, ref Delta, ref WeightDelta, ref BiasDelta);
            }

            for (int i = 1; i < Network.LayerAmount; i++) {
                Network.Layers[i].Weights -= WeightDelta[i - 1] * learning_rate / DataBatch.Size;
                Network.Layers[i].Bias -= BiasDelta[i - 1] * learning_rate / DataBatch.Size;
            }
        }

        public async Task MNIST_Training(int batchSize) {
            MNISTDatabase database = new MNISTDatabase("Assets/StreamingAssets/MNIST/train-images.idx3-ubyte", "Assets/StreamingAssets/MNIST/train-labels.idx1-ubyte");

            UnityEngine.Debug.Log($"Started training on {database.Size} examples.");
            for (int i = 0; i < database.Size; i += batchSize) {
                DataBatch batch = new DataBatch(database.ReadBatch(batchSize));
                BatchTraining(batch);
                UnityEngine.Debug.Log($"Training is {100 * (double)i / database.Size:F2}% Complete [{i}/{database.Size}]");
                await Task.Delay(1);
            }
            UnityEngine.Debug.Log($"Training Complete.");

            database.CloseLoad();
        }
    }
}