using UnityEngine;
using System;
using NeuralNetworkSystem;
using System.Collections.Generic;
using System.Threading.Tasks;

public class ProgramHandler : MonoBehaviour {

    [SerializeField] ForwardFunctionsTypes forward_function = ForwardFunctionsTypes.Parallel;

    [SerializeField] BackwardFunctionsTypes backward_function = BackwardFunctionsTypes.Parallel;
    [SerializeField] BackwardOutputFunctionsTypes backward_output_function = BackwardOutputFunctionsTypes.Parallel;
    [SerializeField] BackwardWeightsFunctionsTypes backward_weight_function = BackwardWeightsFunctionsTypes.Parallel;
    [SerializeField] BackwardBiasFunctionsTypes backward_bias_function = BackwardBiasFunctionsTypes.Parallel;

    [SerializeField] AdjustWeightsFunctionsTypes adjust_weights_function = AdjustWeightsFunctionsTypes.Parallel;
    [SerializeField] AdjustBiasFunctionsTypes adjust_bias_function = AdjustBiasFunctionsTypes.Parallel;

    [SerializeField] InputNormalizationFunctionsType input_normalization = InputNormalizationFunctionsType.NormalizeMeadian;
    [SerializeField] OutputFunctionsType output_activation = OutputFunctionsType.SoftMax;

    [SerializeField] ActivationFunctionsTypes activation_function = ActivationFunctionsTypes.ReLU;
    [SerializeField] LossFunctionsType loss_function = LossFunctionsType.SoftMax;


    public NeuralNetwork Network;
    public NeuralNetworkTrainer Trainer;

    [SerializeField] float learning_rate = 0.075f;
    [SerializeField] int batchSize = 100;
    [SerializeField] int cycles = 1;
    [SerializeField] int seed = 5000;

    public bool disableMessages = true;

    public static ProgramHandler instance;

    private void OnEnable() {
        instance = this;
        CreateNetwork();
    }

    void Update() {
        //if (Input.GetKeyDown(KeyCode.L)) {
        //    if (Network != null) Visualization.Visualize(Network);
        //}
        //if (Input.GetKeyDown(KeyCode.Tab)) Visualization.ToggleInfo();

        if (Input.GetKeyDown(KeyCode.K)) CreateNetwork();
        if (Input.GetKeyDown(KeyCode.M)) Trainer.MNIST_RandomTraining(cycles);
        if (Input.GetKeyDown(KeyCode.Z)) Trainer.ForceStopTraining();
        
        if (Input.GetKeyDown(KeyCode.X)) Trainer.TogglePause();
        if (Input.GetKeyDown(KeyCode.C)) Trainer.ToggleStep();
        if (Input.GetKeyDown(KeyCode.Space)) Trainer.DoStep();

        if (Input.GetKeyDown(KeyCode.V)) {
            DetailVisualization.ClearLosses();
            DetailVisualization.Refresh();
        }
        
        //if (Input.GetKeyDown(KeyCode.Space)) {
        //    if (Input.GetKey(KeyCode.LeftShift)) Test(true);
        //}
        if (Input.GetKeyDown(KeyCode.N)) LargeTest();
        if (Input.GetKeyDown(KeyCode.B)) {
            if (Input.GetKey(KeyCode.LeftShift)) Visualization.instance.WeightDiffToImage();
            else Visualization.instance.WeightToImage();
        }

        if (Input.GetKeyDown(KeyCode.Tab)) disableMessages = !disableMessages;


        SelectCell();
    }

    void SelectCell() {
        if (!CameraHandler.MouseOnScreen()) return;
        if (!Input.GetMouseButtonDown(0)) return;

        Vector2 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);

        RaycastHit2D hit = Physics2D.Raycast(mousePos, Vector2.zero);

        if (hit.collider == null) return;

        ObjectVariables vars = hit.collider.gameObject.GetComponent<ObjectVariables>();
        if (vars == null) return;

        Visualization.Focus(vars.layer, vars.index);
    }

    async Task LargeTest() {
        MNISTDatabase database = new MNISTDatabase("Assets/StreamingAssets/MNIST/t10k-images.idx3-ubyte", "Assets/StreamingAssets/MNIST/t10k-labels.idx1-ubyte");

        List<Data> wrongs = new List<Data>();
        List<int> wrong_labels = new List<int>();

        Debug.Log($"Started testing on {database.Size} test samples.");
        int a = 0;
        for (int i = 0; i < database.Size; i++) {
            Data TestingData = database.ReadBatch(1)[0];
            Vector result = Network.Calculate(TestingData.data);

            int guess = result.MaxIndex();
            if (guess == TestingData.label) {
                a++;
            } else {
                wrongs.Add(TestingData);
                wrong_labels.Add(guess);
            }

            if (i % 1000 == 0) {
                if (!disableMessages) Debug.Log($"Testing is {100 * (double) i / database.Size:F2}% Complete [{i}/{database.Size}]");
                await Task.Delay(1);
            }
        }
        Debug.Log($"Testing complete with {(double)a / database.Size * 100}% accuracy. [{a}/{database.Size}]");
        database.CloseLoad();

        Visualization.instance.DrawImages(wrongs.ToArray(), wrong_labels.ToArray());
    }



    async void CreateNetwork() {
        Debug.Log("New Network created.");
        int[] layers = { 784, 128, 64, 10 };

        UnityEngine.Random.InitState(seed);
        Network = new NeuralNetwork(layers, 
            FunctionManager.GetFunctions(
                forward_function, input_normalization, output_activation,
                backward_function, backward_output_function, backward_weight_function, backward_bias_function,
                adjust_weights_function, adjust_bias_function, activation_function)
            );

        Trainer = new NeuralNetworkTrainer(Network, FunctionManager.GetLossFunction(loss_function), learning_rate, batchSize, cycles);

        await Task.Delay(1);
        DetailVisualization.Initialize(Network, Trainer);
    }
}
