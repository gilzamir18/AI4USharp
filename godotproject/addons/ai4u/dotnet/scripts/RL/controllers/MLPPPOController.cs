using System.Collections;
using System.Collections.Generic;
using Godot;
using ai4u;
using TorchSharp;
using System;

namespace  ai4u;

public partial class MLPPPOController : Controller
{

	[Export]
	public bool tryInitialize = false;

    [Export]
	private string mainOutput = "move";

	private string cmdName = null;
	private float[] fargs = null;
	private int[] iargs = null;
	private bool initialized = false; //indicates if episode has been initialized.

	private ActorCritic model;
	private ModelMetadata metadata; //Metadata of the input and outputs of the agent decision making. 
	private bool isSingleInput = true; //Flag indicating if agent has single sensor or multiple sensors.
	private Dictionary<string, int> inputName2Idx; //mapping sensor name to sensor index.
	private Dictionary<string, float[]> outputs; //mapping model output name to output value.
	private ModelOutput modelOutput; //output metadata.


	private int inputSize;

	private int rewardIdx = -1;
	private int doneIdx = -1;

	private int numberOfSensors = 0;

	override public void OnSetup()
	{		
		BasicAgent bagent = (BasicAgent) agent;
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

		for (int i = 0; i < bagent.Sensors.Count; i++)
		{
			if (bagent.Sensors[i].GetKey() == "reward")
			{
				rewardIdx = i;
			} else if (bagent.Sensors[i].GetKey() == "done")
			{
				doneIdx = i;
			}
			for (int j = 0; j < metadata.inputs.Length; j++)
			{
				if (bagent.Sensors[i].GetName() == metadata.inputs[j].name)
				{
					if (metadata.inputs[j].name == null)
						throw new Exception($"Perception key of the sensor {bagent.Sensors[i].GetType()} cannot be null!");
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
		model = new ActorCritic(inputSize, 32, modelOutput.shape[0]);
		model.Load();
	}
	
	override public void OnReset(Agent agent)
	{

	}


	override public string GetAction()
	{
		if (GetStateAsString(0) == "envcontrol")
		{
			if (GetStateAsString(1).Contains("restart"))
			{
				return ai4u.Utils.ParseAction("__restart__");
			}
			return ai4u.Utils.ParseAction("__noop__");			
		}
		if (cmdName != null && !((BasicAgent)agent).Done )
		{
			if (iargs != null) 
			{
				string cmd = cmdName;
				int[] args = iargs;
				ResetCmd();
				return ai4u.Utils.ParseAction(cmd, args);
			}
			else if (fargs != null)
			{
				string cmd = cmdName;
				float[] args = fargs;
				ResetCmd();
				return ai4u.Utils.ParseAction(cmd, args);
			}
			else
			{
				string cmd = cmdName;
				ResetCmd();
				return ai4u.Utils.ParseAction(cmd);
			}
		}
		else 
		{
			if (initialized)
			{
				if (tryInitialize)
				{
					initialized = true;
					return ai4u.Utils.ParseAction("__restart__");
				}
				return ai4u.Utils.ParseAction("__noop__");
			}
			else
			{
				initialized = true;
				return ai4u.Utils.ParseAction("__restart__");
			}
		}
	}

	override public void NewStateEvent()
	{

		if (GetStateAsString(0) == "envcontrol")
		{
		} else
		{
			var state = GetNextState();
			var action = SelectAction(state.view(-1, inputSize));
			cmdName = mainOutput;
			iargs = new int[]{(int)action.data<long>()[0]};
		}

	}
	
	
    public torch.Tensor SelectAction(torch.Tensor state)
    {
        var actionProbs = model.Act(state);
        var action = torch.multinomial(actionProbs, 1);
        return action;
    }

	private float[] GetInputAsArray(string name)
	{
		return GetStateAsFloatArray(inputName2Idx[name]);
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
	
	private void ResetCmd()
	{
		this.cmdName = null;
		this.iargs = null;
		this.fargs = null;
	}
}

