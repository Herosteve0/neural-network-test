using System;
using System.Diagnostics;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
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

        public DataBatch GetSmallBatch(int index, int size) {
            Data[] newdata = new Data[size];
            Array.Copy(Data, index, newdata, 0, size);
            return new DataBatch(newdata);
        }

        public void Shuffle() {
            for (int i = Size - 1; i > 0; i--) {
                int r = UnityEngine.Random.Range(0, i + 1);
                (Data[i], Data[r]) = (Data[r], Data[i]);
            }
        }
    }

    public class NeuralNetworkTrainer {
        public NeuralNetworkTrainer(NeuralNetwork network, float learning_rate = 0.075f, int batchSize = 100) {
            Network = network;
            this.batchSize = batchSize;
            this.learning_rate = learning_rate;
        }

        NeuralNetwork Network { get; }
        public float learning_rate { get; }
        public int batchSize { get; }


        public static Vector NormalizeInput(Vector v) {
            float mean = 0.1307f;
            float std = 0.3081f;

            Vector R = new Vector(v.Length);
            for (int i = 0; i < v.Length; i++) R[i] = (v[i] - mean) / std;
            return R;
        }


        public static float ReLU(float value) {
            return value > 0 ? value : 0;
        }

        public static float ReLUDerivative(float value) {
            return value > 0 ? 1 : 0;
        }

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

        public void TrainingCalculations(Data TrainingData, ref Matrix[] WeightDelta, ref Vector[] BiasDelta) {
            Vector input = NormalizeInput(TrainingData.data);
            Network.Calculate(input); // all layers of the network have the values we want (inupt, value, activation)

            int length = Network.LayerAmount - 1;

            Vector Y = Vector.SingleValue(Network.LayerLength[length], TrainingData.label);
            float Loss = SoftMaxLoss(Network.Layers[length].Activation, TrainingData.label);
            Vector Cost = Vector.SingleValue(Network.LayerLength[length] , TrainingData.label, Loss);


            Vector tmp;

            int simd_width = Vector<float>.Count;

            for (int i = length; i > 0; i--) {
                Network.Layers[i].Backward(i == length ? Y : Network.Layers[i+1].Values, Network.Layers[i].Delta);

                int j = 0;
                for (; j <= Network.Layers[i].Activation.Transpose())
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

        public async Task MNIST_Training() {
            MNISTDatabase database = new MNISTDatabase("Assets/StreamingAssets/MNIST/train-images.idx3-ubyte", "Assets/StreamingAssets/MNIST/train-labels.idx1-ubyte");

            UnityEngine.Debug.Log($"Started training on {database.Size} examples.");
            int counter = 0;
            for (int i = 0; i < database.Size; i += batchSize) {
                DataBatch batch = new DataBatch(database.ReadBatch(batchSize));
                BatchTraining(batch);
                counter += batchSize;
                if (counter > 100) {
                    UnityEngine.Debug.Log($"Training is {100 * (double)i / database.Size:F2}% Complete [{i}/{database.Size}]");
                    await Task.Delay(1);
                    counter = 0;
                }
            }
            UnityEngine.Debug.Log($"Training Complete.");

            database.CloseLoad();
        }

        public async Task MNIST_RandomTraining(int loops) {
            DataBatch training_data = new DataBatch(MNISTDatabase.LoadAllTrainingData());
            int size = training_data.Size * loops;

            UnityEngine.Debug.Log($"Started training on {size} examples.");
            int delay_counter = 0;
            int counter = 0;
            for (int cycle = 0; cycle < loops; cycle++) {
                training_data.Shuffle();
                for (int i = 0; i < training_data.Size; i += batchSize) {
                    DataBatch batch = training_data.GetSmallBatch(i, batchSize);
                    BatchTraining(batch);
                    counter += batchSize;
                    delay_counter += batchSize;
                    if (delay_counter > 100) {
                        UnityEngine.Debug.Log($"Training is {100 * (double) counter / size:F2}% Complete [{counter}/{size}]");
                        await Task.Delay(1);
                        delay_counter = 0;
                    }
                }
            }
            UnityEngine.Debug.Log($"Training Complete.");
        }
    }
}