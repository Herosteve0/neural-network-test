using UnityEngine;

public enum ObjectType {
    None,
    Neuron,
    Connection
}

public class ObjectVariables : MonoBehaviour {
    public ObjectType type = ObjectType.None;
    public int layer;
    public int index;
    public int index_to;
}
