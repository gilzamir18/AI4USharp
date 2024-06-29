using Godot;
using System;
using System.Collections.Generic;
using TorchSharp;
using static ai4u.math.AI4UMath;
using TorchSharp.Modules;
namespace ai4u;

public partial class MLPPPOTrainerAsync : Trainer
{
	[Export]
	private string mainOutput = "move";

	[Export]
	private MLPPPO model;

	private MLPPPOMemory memory;

	private int horizonPos = 0;
	
	private bool initialized = false; //indicates if episode has been initialized.
	private ModelMetadata metadata; //Metadata of the input and outputs of the agent decision making. 
	private bool isSingleInput = true; //Flag indicating if agent has single sensor or multiple sensors.

	private torch.Tensor state;

	private int policyUpdatesByEpisode = 0;

	private long totalPolicyUpdates = 0;

	private float episodeCriticLoss = 0;
	private float episodePolicyLoss = 0;

	
	private Dictionary<string, int> inputName2Idx; //mapping sensor name to sensor index.
	private Dictionary<string, float[]> outputs; //mapping model output name to output value.
	private ModelOutput modelOutput; //output metadata.

	private int rewardIdx = -1;
	private int doneIdx = -1;

	private int outputSize;
	private int inputSize;

	private long globalSteps = 0;

	private int episodes = -1;

    private MLPPPOAsyncSingleton asyncPlugin;

	public override bool TrainingFinalized()
	{
		return totalPolicyUpdates >=  asyncPlugin.SharedConfig.MaxNumberOfUpdates;
	}

	/// <summary>
	/// Here you allocate extra resources for your specific training loop.
	/// </summary>
	public override void OnSetup()
	{
        asyncPlugin = GetTree().Root.GetNode<MLPPPOAsyncSingleton>("MLPPPOAsyncSingleton");
		if (asyncPlugin == null)
        {
            GD.PrintErr("MLPPPOAsyncSingleton must be configured in your Godot project as an autoload!!!");
        }
		metadata = agent.Metadata;
		outputs = new();
		inputName2Idx = new();
		
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
		model.Build(inputSize, 32, outputSize);
		model.NumberOfEnvs += 1;
		torch.autograd.set_detect_anomaly(true);
	}
	
	///<summary>
	/// Here you get agent life cicle callback about episode resetting.
	///</summary>
	public override void OnReset(Agent agent)
	{
		episodes += 1;
		if (episodes > 0)
		{
			asyncPlugin.Put("endepisode", $"{episodes} {agent.EpisodeReward}");
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
		if ( (horizonPos >= asyncPlugin.SharedConfig.Horizon || !agent.Alive()) && !TrainingFinalized())
		{
			(criticLoss, policyLoss) = model.algorithm.Update(model, memory, true);
            asyncPlugin.Put("workdone", $"{criticLoss} {policyLoss}");
			memory.Clear();
			horizonPos = 0;
			policyUpdatesByEpisode++;
			totalPolicyUpdates++;
		} else if (memory.actions.Count > 0 && ended)
		{
			(criticLoss, policyLoss) = model.algorithm.Update(model, memory, true);
            asyncPlugin.Put("workdone", $"{criticLoss} {policyLoss}");
			horizonPos = 0;
			memory.Clear();
			policyUpdatesByEpisode++;
			totalPolicyUpdates++;
		}
		if (TrainingFinalized())
		{
			asyncPlugin.Put("work finished", "");
		}
		episodeCriticLoss += criticLoss;
		episodePolicyLoss += policyLoss;
		globalSteps++;
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
		var y = model.SelectAction(state.view(-1, inputSize));
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
