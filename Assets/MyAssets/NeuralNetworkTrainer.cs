using System;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;

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
        public NeuralNetworkTrainer(NeuralNetwork network, LossCalculation loss_function, float learning_rate = 0.075f, int batchSize = 100, int cycles = 5) {
            Network = network;
            LossFunction = loss_function;
            this.batchSize = batchSize;
            this.cycles = cycles;
            this.learning_rate = learning_rate;

            isTraining = false;
            PausedTraining = false;
            StepTraining = false;
        }

        NeuralNetwork Network { get; }
        LossCalculation LossFunction { get; }
        public float learning_rate { get; }
        public int batchSize { get; }
        public int cycles { get; }
        public int Seed { get; }

        public bool isTraining { get; private set; }
        public bool PausedTraining { get; private set; }
        public bool StepTraining { get; private set; }
        public int TrainingProgress { get; private set; }
        public int TrainingAmount { get; private set; }

        CancellationTokenSource canceltoken;
        

        public float TrainingCalculations(Data TrainingData, ref Matrix[] WeightDelta, ref Vector[] BiasDelta) {
            Vector output = Network.Calculate(TrainingData.data); // all layers of the network have the values we want (inupt, value, activation)

            int length = Network.LayerAmount - 1;

            Vector Y = Vector.SingleValue(Network.LayerLength[length], TrainingData.label);
            float Loss = LossFunction(output.Data, TrainingData.label);

            int simd_width = Vector<float>.Count;

            for (int i = length; i > 0; i--) {
                int l = i - 1;
                
                Network.Layers[i].Backward(Y); // input only used in output layer
                Network.Layers[i].BackwardWeights(ref WeightDelta[l]);
                Network.Layers[i].BackwardBias(ref BiasDelta[l]);
            }

            return Loss;
        }

        public async Task SingleExampleTraining(Data TrainingData) {
            await BatchTraining(new DataBatch(new Data[] { TrainingData }));
        }
        public async Task BatchTraining(DataBatch DataBatch) {
            Vector[] Delta = new Vector[Network.LayerAmount - 1];
            Matrix[] WeightDelta = new Matrix[Network.LayerAmount - 1];
            Vector[] BiasDelta = new Vector[Network.LayerAmount - 1];

            for (int i = 0; i < Network.LayerAmount - 1; i++) {
                WeightDelta[i] = new Matrix(Network.LayerLength[i + 1], Network.LayerLength[i]);
                BiasDelta[i] = new Vector(Network.LayerLength[i + 1]);
            }

            float avg = 0f;

            foreach (Data TrainingData in DataBatch.Data) {
                if (StepTraining) {
                    canceltoken = new CancellationTokenSource();
                    await WaitFor(-1, canceltoken.Token);
                }
                avg += TrainingCalculations(TrainingData, ref WeightDelta, ref BiasDelta);
            }
            avg /= DataBatch.Size;
            DetailVisualization.StoreLoss(avg);

            float scale = learning_rate / DataBatch.Size;
            for (int i = 1; i < Network.LayerAmount; i++) {
                int l = i - 1;
                Network.Layers[i].AdjustWeight(ref WeightDelta[l], scale);
                Network.Layers[i].AdjustBias(ref BiasDelta[l], scale);
            }
        }


        public void ForceStopTraining() {
            if (!isTraining) return;
            isTraining = false;
            PrintMessage(ConsoleMessages.ForceStop);
        }
        public void TogglePause() {
            PausedTraining = !PausedTraining;
            if (PausedTraining) canceltoken = new CancellationTokenSource();
            else canceltoken.Cancel();
            PrintMessage(ConsoleMessages.Pause);
        }
        public void ToggleStep() {
            StepTraining = !StepTraining;
            if (StepTraining) canceltoken = new CancellationTokenSource();
            else canceltoken.Cancel();
            PrintMessage(ConsoleMessages.Step);
        }
        public void DoStep() {
            if (!StepTraining) return;
            PrintMessage(ConsoleMessages.DoStep);
            canceltoken.Cancel();
        }

        enum ConsoleMessages {
            Start,
            Progress,
            Finish,
            ForceStop,
            Pause,
            Step,
            DoStep,
        }
        void PrintMessage(ConsoleMessages type) {
            if (type == ConsoleMessages.Start) UnityEngine.Debug.Log($"Started training on {TrainingAmount} examples.");
            else if (type == ConsoleMessages.Progress) {
                if (ProgramHandler.instance.disableMessages) return;
                UnityEngine.Debug.Log($"Training is {100 * (double)TrainingProgress / TrainingAmount:F2}% Complete [{TrainingProgress}/{TrainingAmount}]");
            } else if (type == ConsoleMessages.Finish) UnityEngine.Debug.Log($"Training Complete.");
            else if (type == ConsoleMessages.ForceStop) UnityEngine.Debug.Log($"Force stopped training.");
            else if (type == ConsoleMessages.Pause) UnityEngine.Debug.Log((PausedTraining ? "Paused" : "Unpaused") + " training.");
            else if (type == ConsoleMessages.Step) UnityEngine.Debug.Log((StepTraining ? "Enabled" : "Disabled") + " step training.");
            else if (type == ConsoleMessages.DoStep) UnityEngine.Debug.Log("Did one training step.");
        }

        int delay_ticks = 750;

        //public async Task MNIST_Training() {
        //    MNISTDatabase database = new MNISTDatabase("Assets/StreamingAssets/MNIST/train-images.idx3-ubyte", "Assets/StreamingAssets/MNIST/train-labels.idx1-ubyte");

        //    TrainingProgress = 0;
        //    TrainingAmount = database.Size;
        //    isTraining = true;

        //    PrintMessage(ConsoleMessages.Start);
        //    int counter = 0;
        //    for (int i = 0; i < database.Size; i += batchSize) {
        //        if (!isTraining) return;
        //        bool breath = counter >= delay_ticks;
        //        await Train(new DataBatch(database.ReadBatch(batchSize)), breath);
        //        if (breath) counter = 0;
        //    }
        //    isTraining = false;
        //    database.CloseLoad();

        //    PrintMessage(ConsoleMessages.Finish);
        //    DetailVisualization.Refresh();
        //}



        async Task Train(DataBatch batch, bool breath) {
            if (PausedTraining) await WaitFor(-1, canceltoken.Token);

            await BatchTraining(batch);
            TrainingProgress += batchSize;
            if (breath) {
                PrintMessage(ConsoleMessages.Progress);
                DetailVisualization.Refresh();
                await Task.Delay(1);
            }

        }

        async Task WaitFor(int ms, CancellationToken token) {
            try {
                await Task.Delay(ms, token);
            } catch (TaskCanceledException) { }
        }

        public async Task MNIST_RandomTraining() { await MNIST_RandomTraining(cycles); }
        public async Task MNIST_RandomTraining(int loops) {
            DataBatch training_data = new DataBatch(MNISTDatabase.LoadAllTrainingData());

            TrainingProgress = 0;
            TrainingAmount = training_data.Size * loops;
            isTraining = true;

            PrintMessage(ConsoleMessages.Start);
            int counter = 0;
            for (int cycle = 0; cycle < loops; cycle++) {
                training_data.Shuffle();
                for (int i = 0; i < training_data.Size; i += batchSize) {
                    if (!isTraining) return;

                    counter += batchSize;
                    bool breath = counter >= delay_ticks;
                    await Train(training_data.GetSmallBatch(i, batchSize), breath);
                    if (breath) counter = 0;
                }
            }
            isTraining = false;
            
            PrintMessage(ConsoleMessages.Finish);
            DetailVisualization.Refresh();
        }
    }
}