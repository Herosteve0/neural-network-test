using UnityEngine;
using NeuralNetworkSystem;
using System.Threading.Tasks;
using System;

public class ProgramHandler : MonoBehaviour {

    [SerializeField] ForwardFunctionsTypes forward_function = ForwardFunctionsTypes.SIMD;

    [SerializeField] BackwardFunctionsTypes backward_function = BackwardFunctionsTypes.SIMD;
    [SerializeField] BackwardOutputFunctionsTypes backward_output_function = BackwardOutputFunctionsTypes.SIMD;
    [SerializeField] BackwardWeightsFunctionsTypes backward_weight_function = BackwardWeightsFunctionsTypes.SIMD;
    [SerializeField] BackwardBiasFunctionsTypes backward_bias_function = BackwardBiasFunctionsTypes.SIMD;

    [SerializeField] AdjustWeightsFunctionsTypes adjust_weights_function = AdjustWeightsFunctionsTypes.SIMD;
    [SerializeField] AdjustBiasFunctionsTypes adjust_bias_function = AdjustBiasFunctionsTypes.SIMD;

    [SerializeField] InputNormalizationFunctionsType input_normalization = InputNormalizationFunctionsType.NormalizeMeadian;
    [SerializeField] OutputFunctionsType output_activation = OutputFunctionsType.SoftMax;

    [SerializeField] ActivationFunctionsTypes activation_function = ActivationFunctionsTypes.ReLU;
    [SerializeField] LossFunctionsType loss_function = LossFunctionsType.SoftMax;


    public NeuralNetwork Network;
    public NeuralNetworkTrainer Trainer;

    [SerializeField] int[] hidden_layers;
    [SerializeField] float learning_rate = 0.075f;
    [SerializeField] bool learning_rate_decay = true;
    [SerializeField] int learning_rate_decay_patience = 500;
    [SerializeField] int batchSize = 100;
    [SerializeField] int cycles = 1;
    [SerializeField] int seed = 5000;

    public bool disableMessages = true;

    public static ProgramHandler instance;
    public static int version = 0;

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

        if (Input.GetKeyDown(KeyCode.A)) {
            Data data = MNISTDatabase.LoadAllTrainingData()[0];
            Trainer.SingleExampleTraining(data);
            DetailVisualization.Refresh();
        }

        if (Input.GetKeyDown(KeyCode.V)) {
            DetailVisualization.ClearLosses();
            DetailVisualization.Refresh();
        }

        if (Input.GetKeyDown(KeyCode.Q)) {
            NeuralNetworkStoring.Save(Network, "Assets/StreamingAssets/Save.nn");
            Debug.Log("Saved Network!");
        }
        if (Input.GetKeyDown(KeyCode.W)) {
            CreateNetwork(NeuralNetworkStoring.Load("Assets/StreamingAssets/Save.nn"));
        }
        
        //if (Input.GetKeyDown(KeyCode.Space)) {
        //    if (Input.GetKey(KeyCode.LeftShift)) Test(true);
        //}
        if (Input.GetKeyDown(KeyCode.N)) Trainer.MINST_Test();
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


    public static LayerFunctions GetNetworkFunctions() {
        return FunctionManager.GetFunctions(
                instance.forward_function, instance.input_normalization, instance.output_activation,
                instance.backward_function, instance.backward_output_function, instance.backward_weight_function, instance.backward_bias_function,
                instance.adjust_weights_function, instance.adjust_bias_function, instance.activation_function);
    }
    async void CreateNetwork() {
        Debug.Log("New Network created.");
        int[] layers = new int[hidden_layers.Length + 2];
        layers[0] = 784;
        for (int i = 0; i < hidden_layers.Length; i++) {
            layers[i + 1] = hidden_layers[i];
        }
        layers[^1] = 10;

        if (Network != null) Trainer.ForceStopTraining();

        UnityEngine.Random.InitState(seed);
        Network = new NeuralNetwork(layers, GetNetworkFunctions());

        Trainer = new NeuralNetworkTrainer(Network, FunctionManager.GetLossFunction(loss_function), learning_rate, learning_rate_decay, learning_rate_decay_patience, batchSize, cycles);

        await Task.Delay(1);
        DetailVisualization.Initialize(Network, Trainer);
    }
    async void CreateNetwork(NeuralNetwork network) {
        Debug.Log("Loaded Network.");

        if (Network != null) Trainer.ForceStopTraining();
        Network = network;

        UnityEngine.Random.InitState(seed);
        Trainer = new NeuralNetworkTrainer(Network, FunctionManager.GetLossFunction(loss_function), learning_rate, learning_rate_decay, learning_rate_decay_patience, batchSize, cycles);

        await Task.Delay(1);
        DetailVisualization.Initialize(Network, Trainer);
    }
}
