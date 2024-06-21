using Godot;
using System;
using System.Collections.Generic;
using TorchSharp;
using static ai4u.math.AI4UMath;
using TorchSharp.Utils.tensorboard;
using TorchSharp.Modules;
namespace ai4u;

public partial class MLPPPOTrainer : Trainer
{
	private MLPPPO ppoAlg;

	private int inputSize = -1;

	private int numberOfSensors = -1;

    /// <summary>
    /// The name of the actuator that receives commands from the neural network 
	/// in the ONNX model.
    /// </summary>
    [Export]
	private string mainOutput = "move";
	
	/// <summary>
	/// If true, episode is restarted after ending.
	/// </summary>
	[Export]
	private bool repeat = true;
	[Export]
	private float learningRate = 0.00025f;
	[Export]
	private float gamma = 0.99f;
	[Export]
	private float lambda = 0.95f;
	[Export]
	private float clipParam = 0.2f;
	[Export]
	private int horizon = 5;
	[Export]
	private int updates = 1;

	[Export]
	private int maxNumberOfUpdates = 1000;

	[Export]
	private string logPath = "";

	private MLPPPOMemory memory;

	private int horizonPos = 0;

	
	private bool initialized = false; //indicates if episode has been initialized.
	private ModelMetadata metadata; //Metadata of the input and outputs of the agent decision making. 
	private bool isSingleInput = true; //Flag indicating if agent has single sensor or multiple sensors.
	private Dictionary<string, int> inputName2Idx; //mapping sensor name to sensor index.
	private Dictionary<string, float[]> outputs; //mapping model output name to output value.
	private ModelOutput modelOutput; //output metadata.

	private int rewardIdx = -1;
	private int doneIdx = -1;

	private torch.Tensor state;

	private int policyUpdatesByEpisode = 0;

	private long totalPolicyUpdates = 0;

	private float episodeCriticLoss = 0;
	private float episodePolicyLoss = 0;

	private bool modelSaved = false;

	private SummaryWriter summaryWriter;
	

	public override bool TrainingFinalized()
	{
		return totalPolicyUpdates >=  maxNumberOfUpdates;
	}

	/// <summary>
	/// Here you allocate extra resources for your specific training loop.
	/// </summary>
	public override void OnSetup()
	{
		summaryWriter = torch.utils.tensorboard.SummaryWriter(logPath, "_log");
		inputName2Idx = new Dictionary<string, int>();
		outputs = new Dictionary<string, float[]>();
		metadata = agent.Metadata;
		for (int o = 0; o < metadata.outputs.Length; o++)
		{
			var output = metadata.outputs[o];
			outputs[output.name] = new float[output.shape[0]];
			if (output.name == mainOutput)
			{
				modelOutput = output;
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
		memory = new();
		ppoAlg = new MLPPPO(inputSize, 32, modelOutput.shape[0], clipParam, gamma, lambda, learningRate);
		ppoAlg.Updates = updates;
		modelSaved = false;
	}	
	
	///<summary>
	/// Here you get agent life cicle callback about episode resetting.
	///</summary>
	public override void OnReset(Agent agent)
	{
		GD.Print("Episode Reward: " + agent.EpisodeReward);
		summaryWriter.add_scalar("episode/reward", agent.EpisodeReward, (int)totalPolicyUpdates);
		GD.Print("Updates: " + totalPolicyUpdates);
		if (policyUpdatesByEpisode > 0)
		{
			GD.Print("Critic Loss: " + episodeCriticLoss/policyUpdatesByEpisode);
			GD.Print("Policy Loss: " + episodePolicyLoss/policyUpdatesByEpisode);
		}
		policyUpdatesByEpisode = 0;
		ended = false;
		horizonPos = 0;
		memory.Clear();
		state = GetNextState();
		episodeCriticLoss = 0;
		episodePolicyLoss = 0;
	}

	/// <summary>
	/// This callback method run after agent percept a new state.
	/// </summary>
	public override void StateUpdated()
	{
		if (ended || !agent.SetupIsDone)
		{
			GD.Print("End of episode!");
			return;
		}
		CollectData();
		float criticLoss = 0;
		float policyLoss = 0;
		if ( (horizonPos >= horizon || !agent.Alive()) && !TrainingFinalized())
		{
			(criticLoss, policyLoss) = ppoAlg.Update(memory);
			summaryWriter.add_scalar("critic/loss", criticLoss, (int)totalPolicyUpdates);
			summaryWriter.add_scalar("policy/loss", policyLoss, (int)totalPolicyUpdates);
			memory.Clear();
			policyUpdatesByEpisode++;
			totalPolicyUpdates++;
			if (!modelSaved && TrainingFinalized())
			{
				modelSaved = true;
				ppoAlg.Save();
			}
		} else if (memory.actions.Count > 0 && ended)
		{
			(criticLoss, policyLoss) = ppoAlg.Update(memory);
			memory.Clear();
			policyUpdatesByEpisode++;
			totalPolicyUpdates++;
		}
		episodeCriticLoss += criticLoss;
		episodePolicyLoss += policyLoss;
	}
	/// <summary>
	/// This method gets state from sensor named <code>name</code> and returns its value as an array of float-point numbers.
	/// </summary>
	/// <param name="name"></param>
	/// <returns>float[]: sensor value</returns>
	private float[] GetInputAsArray(string name)
	{
		return controller.GetStateAsFloatArray(inputName2Idx[name]);
	}


	public override void EnvironmentMessage()
	{
		
	}

	private static torch.Tensor oldLogProbs = null, oldValues = null;

	private static bool ended = false;

	private void CollectData()
	{
		var reward = controller.GetStateAsFloat(rewardIdx);
		var done = controller.GetStateAsBool(doneIdx);

		var nextState = GetNextState();
		if ( state is null)
		{
			state = GetNextState();
		}
		var y = ppoAlg.SelectAction(state.view(-1, inputSize));
		long action = y.data<long>()[0];

		controller.RequestAction(mainOutput, new int[]{ (int)action});
		
		memory.actions.Add(y.detach());
		memory.states.Add(state.detach());
		memory.rewards.Add(reward);
		memory.dones.Add(done);
		horizonPos++;
		ended = done;
		state = nextState;
	}

	private torch.Tensor GetNextState()
	{
		torch.Tensor t = null;

		for (int i = 0; i < metadata.inputs.Length; i++)
		{
			var inputName = metadata.inputs[i].name;
			var shape = metadata.inputs[i].shape;
			var dataDim = shape.Length;
			
			if (dataDim  == 1 && metadata.inputs[i].type == SensorType.sfloatarray)
			{ 
				var fvalues = GetInputAsArray(inputName);
				
				t = torch.FloatTensor(fvalues);
				//t = t.reshape(new long[2]{shape[0]});
			}
			else
			{
				throw new System.Exception($"Controller configuration Error: for while, only MLPPolicy is supported!");
			}
		}
		return t;
	}
}
