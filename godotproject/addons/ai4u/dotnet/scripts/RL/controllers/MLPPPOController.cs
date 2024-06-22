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

	private string cmdName = null;
	private float[] fargs = null;
	private int[] iargs = null;
	private bool initialized = false; //indicates if episode has been initialized.

	private ModelMetadata metadata; //Metadata of the input and outputs of the agent decision making. 


	override public void OnSetup()
	{		
		metadata = agent.Metadata;
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

		if (GetStateAsString(0) == "envcontrol")
		{
		} else
		{
			var state = GetNextState();
			var action = model.SelectAction(state.view(-1, model.InputSize));
			cmdName = model.MainOutputName;
			iargs = new int[]{(int)action.data<long>()[0]};
		}

	}
	

	private float[] GetInputAsArray(string name)
	{
		return GetStateAsFloatArray( model.GetInputIdx(name) );
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

