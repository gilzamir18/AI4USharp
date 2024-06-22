using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using System.Linq;
using Godot;

namespace ai4u;

public class MLPPPOMemory
{
    public List<Tensor> states = new List<Tensor>();
    public List<Tensor> actions = new List<Tensor>();
    public List<float> rewards = new List<float>();
    public List<bool> dones = new List<bool>();

    public void Clear()
    {
        states.Clear();
        actions.Clear();
        rewards.Clear();
        dones.Clear();
    }
}

public class ActorCritic : Module
{
    private readonly Sequential actor;
    private readonly Sequential critic;

    public Sequential Actor => actor;
    public Sequential Critic => critic;

    public ActorCritic(int inputSize, int hiddenSize, int actionSize) : base(nameof(ActorCritic))
    {
        actor = Sequential(
            Linear(inputSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, actionSize),
            Softmax(1)
        );

        critic = Sequential(
            Linear(inputSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, 1)
        );

        RegisterComponents();
    }

    public Tensor Act(Tensor input)
    {
        return actor.forward(input);
    }

    public Tensor Evaluate(Tensor input)
    {
        return critic.forward(input);
    }

    public void Save(string prefix="model", string path="")
    {
        GD.Print(System.IO.Path.Join(path, prefix + "_critic.dat"));
        actor.save(System.IO.Path.Join(path, prefix + "_actor.dat"));
        critic.save(System.IO.Path.Join(path, prefix + "_critic.dat"));
    }

    public void Load(string prefix="model", string path="")
    {
        actor.load (System.IO.Path.Join(path, prefix + "_actor.dat"));
        critic.load(System.IO.Path.Join(path, prefix + "_critic.dat"));
    }
}


public partial class MLPPPO: Node
{
    internal ActorCritic policy;
    internal ActorCritic oldPolicy;
    internal torch.optim.Optimizer optimizer;

    [Export]
    private Agent agent;

    [Export]
	private string mainOutput = "move";

    [ExportGroup("Model Size")]
    [Export]
    private int hiddenSize = 32;

    [ExportGroup("Training Mode")]
    internal MLPPPOAlgorithm algorithm;

    [ExportGroup("Metadata Info")]
    [Export]
    private int inputSize = 2;
    [Export]
    private int outputSize = 4;

    [ExportGroup("Async Case")]
    [Export]
    private bool shared = false;
    [Export]
    public int SyncFrequence {get; set;} = 1;


	private bool initialized = false; //indicates if episode has been initialized.
	private ModelMetadata metadata; //Metadata of the input and outputs of the agent decision making. 
	private bool isSingleInput = true; //Flag indicating if agent has single sensor or multiple sensors.
	private Dictionary<string, int> inputName2Idx; //mapping sensor name to sensor index.
	private Dictionary<string, float[]> outputs; //mapping model output name to output value.
	private ModelOutput modelOutput; //output metadata.

	private int rewardIdx = -1;
	private int doneIdx = -1;


    public int RewardIndex => rewardIdx;
    public int DoneIndex => doneIdx;


    public int GetInputIdx (string name) => inputName2Idx[name];  

    public string MainOutputName => mainOutput;

    public int InputSize => inputSize;

	private int numberOfSensors = -1;

    private bool dataLoaded = false;


    private delegate void OnModelDataLoaded();

    private event OnModelDataLoaded OnDataLoaded;

    public override void _Ready()
    {

        if (agent == null && GetParent() is Agent)
        {
            agent = GetParent() as Agent;
        }

        if (shared)
        {
            if (agent == null)
            {        
                policy = new ActorCritic(inputSize, hiddenSize, outputSize);
                oldPolicy = new ActorCritic(inputSize, hiddenSize, outputSize);
            }

            if (algorithm != null)
            {
                optimizer = torch.optim.Adam(policy.parameters(), algorithm.LearningRate);
            }
            else
            {
                optimizer = torch.optim.Adam(policy.parameters(), 0.00025f);
                GD.PrintRich("Warning: this model is in shared mode, so it must have an associated MLPPPOAlgoritm!");
            }
            dataLoaded = true;
            GetTree().Root.GetNode<MLPPPOAsyncSingleton>("MLPPPOAsyncPlugin").Model = this;
        }


        if (agent != null)
        {
            if (agent.SetupIsDone)
            {
                LoadData(agent);
            }
            else
            {
                agent.OnSetupDone += LoadData;
            }
        }
    }

    private void LoadData(Agent agent)
    {
		metadata = agent.Metadata;
        outputs = new Dictionary<string, float[]>();
		inputName2Idx = new Dictionary<string, int>();

    	for (int o = 0; o < metadata.outputs.Length; o++)
		{
			var output = metadata.outputs[o];
			outputs[output.name] = new float[output.shape[0]];
			if (output.name == mainOutput)
			{
				modelOutput = output;
                outputSize = output.shape[0];
			}
		}

		for (int i = 0; i < agent.Sensors.Count; i++)
		{
			if (agent.Sensors[i].GetKey() == "reward")
			{
				rewardIdx = i;
			} else if (agent.Sensors[i].GetKey() == "done")
			{
				doneIdx = i;
			}
			for (int j = 0; j < metadata.inputs.Length; j++)
			{
				if (agent.Sensors[i].GetName() == metadata.inputs[j].name)
				{
					if (metadata.inputs[j].name == null)
						throw new Exception($"Perception key of the sensor {agent.Sensors[i].GetType()} cannot be null!");
					inputName2Idx[metadata.inputs[j].name] = i;
					inputSize = metadata.inputs[i].shape[0];
					numberOfSensors ++;	
				}
			}
		}

		if (metadata.inputs.Length == 1)
		{
			isSingleInput = true;
		}
		else
		{
			isSingleInput = false;
			throw new System.Exception("Only one input is supported!!!");
		}
        
        policy = new ActorCritic(inputSize, hiddenSize, outputSize);
        oldPolicy = new ActorCritic(inputSize, hiddenSize, outputSize);
        if (algorithm != null && optimizer == null)
        {
            GD.PrintRich("Warning: this model is in training mode!");
            optimizer = torch.optim.Adam(policy.parameters(), algorithm.LearningRate);
        }
        dataLoaded = true;
    }

    public void Save()
    {
        policy.Save();
    }

    public void SyncWith(MLPPPO model)
    {
        policy.Actor.load_state_dict(model.policy.Actor.state_dict());
        policy.Critic.load_state_dict(model.policy.Critic.state_dict());
        oldPolicy.Actor.load_state_dict(model.oldPolicy.Actor.state_dict());
        oldPolicy.Critic.load_state_dict(model.oldPolicy.Critic.state_dict());
    }

    public Tensor SelectAction(Tensor state)
    {
        var actionProbs = policy.Act(state);
        var action = torch.multinomial(actionProbs, 1);
        return action;
    }

    public void Load(string prefix="model", string path="")
    {
        if (dataLoaded)
        {
            policy.Load(prefix, path);
            oldPolicy.Load(prefix, path);
        }
        else
        {
            OnDataLoaded += () => {
                policy.Load(prefix, path);
                oldPolicy.Load(prefix, path);
            };
        }
    }
}
