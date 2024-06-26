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
	public MLPPPO model;
	[Export]
	private string mainOutput = "move";
	private string cmdName = null;
	private float[] fargs = null;
	private int[] iargs = null;
	private bool initialized = false; //indicates if episode has been initialized.

	private bool isSingleInput = false;

	private ModelMetadata metadata; //Metadata of the input and outputs of the agent decision making. 

	private Dictionary<string, int> inputName2Idx; //mapping sensor name to sensor index.
	private Dictionary<string, float[]> outputs; //mapping model output name to output value.
	private ModelOutput modelOutput; //output metadata.
	
	private int rewardIdx = -1;
	private int doneIdx = -1;

	private int outputSize;
	private int inputSize;

	private long globalSteps = 0;
	override public void OnSetup()
	{		
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
		model.Build(inputSize, 32, outputSize);
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
		if (cmdName != null && !agent.Done )
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

		if (GetStateAsString(0) != "envcontrol")
		{
			var state = GetNextState();
			var action = model.SelectAction(state.view(-1, model.InputSize));
			cmdName = mainOutput;
			iargs = new int[]{(int)action.data<long>()[0]};
		}
	}
	

	private float[] GetInputAsArray(string name)
	{
		return GetStateAsFloatArray( inputName2Idx[name] );
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

