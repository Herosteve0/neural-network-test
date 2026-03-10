using System;
using System.Collections.Generic;
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
        public NeuralNetworkTrainer(NeuralNetwork network, LossCalculation loss_function, float learning_rate = 0.075f, bool lr_decay = true, int lr_decay_patience = 10, int batchSize = 100, int cycles = 5) {
            Network = network;
            LossFunction = loss_function;
            this.batchSize = batchSize;
            this.cycles = cycles;
            base_learning_rate = learning_rate;
            LearningRate = base_learning_rate;
            learning_rate_decay = lr_decay;
            learning_rate_decay_patience = lr_decay_patience;

            timeDelta = 0;
            isTraining = false;
            isTesting = false;
            PausedTraining = false;
            StepTraining = false;
        }

        NeuralNetwork Network { get; }
        LossCalculation LossFunction { get; }
        
        public float base_learning_rate { get; }
        public float LearningRate { get; private set; }
        public bool learning_rate_decay {  get; }
        public int learning_rate_decay_patience {  get; }
        float min_loss = -1;
        int loss_counter = 0;

        public int batchSize { get; }
        public int cycles { get; }
        public int Seed { get; }

        public bool isTraining { get; private set; }
        public bool PausedTraining { get; private set; }
        public bool StepTraining { get; private set; }
        public int TrainingProgress { get; private set; }
        public int TrainingAmount { get; private set; }

        public bool isTesting { get; private set; }
        public int TestingProgress { get; private set; }
        public int TestingAccuracy { get; private set; }
        public int TestingAmount { get; private set; }

        public double timeDelta { get; private set; }
        DateTime timeTemp;

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

            if (learning_rate_decay) {
                if (min_loss == -1f) min_loss = avg;
                if (avg >= min_loss) {
                    loss_counter++;
                    if (loss_counter > learning_rate_decay_patience) {
                        loss_counter = 0;
                        LearningRate *= 0.5f;
                    }
                } else {
                    min_loss = avg;
                    loss_counter = 0;
                }
            }

            float scale = LearningRate / DataBatch.Size;
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

            TestStart,
            TestProgress,
            TestFinish,

            ErrorTraining,
            ErrorTesting
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

            else if (type == ConsoleMessages.TestStart) { UnityEngine.Debug.Log($"Started testing on {TestingAmount} test samples."); } else if (type == ConsoleMessages.TestProgress) {
                if (ProgramHandler.instance.disableMessages) return;
                UnityEngine.Debug.Log($"Testing is {100 * (double)TestingProgress / TestingAmount:F2}% Complete [{TestingProgress}/{TestingAmount}]");
            } else if (type == ConsoleMessages.TestFinish) {
                UnityEngine.Debug.Log($"Testing complete with {(double)TestingAccuracy / TestingAmount * 100}% accuracy. [{TestingAccuracy}/{TestingAmount}]");

            } else if (type == ConsoleMessages.ErrorTraining) UnityEngine.Debug.Log($"Wait until the training is complete before starting the testing process.");
            else if (type == ConsoleMessages.ErrorTesting) { UnityEngine.Debug.Log($"Wait until the testing is complete before starting the training process."); }    
        }

        int delay_ticks = 750;

        async Task Train(DataBatch batch, bool breath) {
            if (PausedTraining) await WaitFor(-1, canceltoken.Token);

            await BatchTraining(batch);
            TrainingProgress += batchSize;
            if (breath) {
                PrintMessage(ConsoleMessages.Progress);
                DetailVisualization.Refresh();
                timeDelta = (DateTime.Now - timeTemp).TotalSeconds;
                await Task.Delay(1);
            }
            timeTemp = DateTime.Now;

        }

        async Task WaitFor(int ms, CancellationToken token) {
            try {
                await Task.Delay(ms, token);
            } catch (TaskCanceledException) { }
        }

        public async Task MNIST_RandomTraining() { await MNIST_RandomTraining(cycles); }
        public async Task MNIST_RandomTraining(int loops) {
            if (isTesting) {
                PrintMessage(ConsoleMessages.ErrorTesting);
                return;
            }
            DataBatch training_data = new DataBatch(MNISTDatabase.LoadAllTrainingData());

            TrainingProgress = 0;
            TrainingAmount = training_data.Size * loops;
            isTraining = true;

            PrintMessage(ConsoleMessages.Start);
            int counter = 0;
            timeTemp = DateTime.Now;
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


        public async Task MINST_Test() {
            if (isTraining) {
                PrintMessage(ConsoleMessages.ErrorTraining);
                return;
            }
            MNISTDatabase database = new MNISTDatabase("Assets/StreamingAssets/MNIST/t10k-images.idx3-ubyte", "Assets/StreamingAssets/MNIST/t10k-labels.idx1-ubyte");

            List<Data> wrongs = new List<Data>();
            List<int> wrong_labels = new List<int>();

            isTesting = true;
            TestingAmount = database.Size;
            TestingAccuracy = 0;
            PrintMessage(ConsoleMessages.TestStart);
            int counter = 0;
            timeTemp = DateTime.Now;
            for (TestingProgress = 0; TestingProgress < TestingAmount; TestingProgress++) {
                Data TestingData = database.ReadBatch(1)[0];
                Vector result = Network.Calculate(TestingData.data);

                int guess = result.MaxIndex();
                if (guess == TestingData.label) {
                    TestingAccuracy++;
                } else {
                    wrongs.Add(TestingData);
                    wrong_labels.Add(guess);
                }

                counter += 1;
                if (counter >= delay_ticks) {
                    PrintMessage(ConsoleMessages.TestProgress);
                    DetailVisualization.Refresh();
                    timeDelta = (DateTime.Now - timeTemp).TotalSeconds;
                    await Task.Delay(1);
                    counter = 0;
                }
                timeTemp = DateTime.Now;
            }
            isTesting = false;
            database.CloseLoad();

            PrintMessage(ConsoleMessages.TestFinish);
            DetailVisualization.Refresh();
            Visualization.instance.DrawImages(wrongs.ToArray(), wrong_labels.ToArray());
        }
    }
}