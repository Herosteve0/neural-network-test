using UnityEngine;

using NeuralNetworkSystem;
using System.Collections.Generic;
using System.Threading.Tasks;

public class ProgramHandler : MonoBehaviour {
    
    public NeuralNetwork Network;
    public NeuralNetworkTrainer Trainer;
    
    [SerializeField] float learning_rate = 0.075f;
    [SerializeField] int batchSize = 100;
    [SerializeField] int cycles = 5;
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
        if (Input.GetKeyDown(KeyCode.S)) {
            for (int i = 0; i < 500; i++) DetailVisualization.StoreLoss(i / 125f);
            DetailVisualization.Refresh();
        }
        if (Input.GetKeyDown(KeyCode.D)) {
            Vector a = new Vector(
                new float[] { -0.3f, 0.1f, 0.7f, -0.6f }
            );

            Debug.Log("");
            Data data = new Data(a, 0);
            Trainer.SingleExampleTraining(data);

            //for (int i = 0; i < Network.LayerAmount; i++) {
            //    Debug.Log($"Layer {i}:\n Values {Network.Layers[i].Values}\n Activation {Network.Layers[i].Activation}");
            //}
        }

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

        Debug.Log($"Started testing on {database.Size} test samples.");
        int a = 0;
        for (int i = 0; i < database.Size; i++) {
            Data TestingData = database.ReadBatch(1)[0];
            Vector result = Network.Calculate(TestingData.data);
            if (result.MaxIndex() == TestingData.label) a++;
            if (i % 1000 == 0) {
                Debug.Log($"Testing is {100 * (double) i / database.Size:F2}% Complete [{i}/{database.Size}]");
                await Task.Delay(1);
            }
        }
        Debug.Log($"Testing complete with {(double)a / database.Size * 100}% accuracy. [{a}/{database.Size}]");
        database.CloseLoad();
    }



    async void CreateNetwork() {
        Debug.Log("New Network created.");
        int[] layers = { 784, 128, 64, 10 };
        //int[] layers = { 4, 3, 2 };

        UnityEngine.Random.InitState(seed);
        Network = new NeuralNetwork(layers);

        Trainer = new NeuralNetworkTrainer(Network, learning_rate, batchSize, cycles);

        await Task.Delay(1);
        DetailVisualization.Initialize(Network, Trainer);
    }
}
