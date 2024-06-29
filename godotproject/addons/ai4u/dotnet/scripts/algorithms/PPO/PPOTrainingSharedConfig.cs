using ai4u;
using Godot;
using System;

public partial class PPOTrainingSharedConfig : Node
{
	[Export]
	public string MainOutput {get;set;} = "move";
	[Export(PropertyHint.Range, "1,100000,or_greater")]
	public int Horizon {get; set;} = 32;
	[Export(PropertyHint.Range, "10,100000000,or_greater")]
	public int MaxNumberOfUpdates {get; set;} = 2000;
	[Export(PropertyHint.Range, "1,100")]
    public int UpdateGradientInterval {get; set;} = 1;

    public override void _EnterTree()
    {
        base._EnterTree();
		var p = GetTree().Root.GetNode("MLPPPOAsyncSingleton");

		if (p == null)
		{
			GD.PrintErr(@"Error: You added a TrainingSharedConfig to your scene, 
			but there isn't any Autoload named MLPPPOAsyncSingleton of the type ai4u.MLPPPOAsyncSingleton. 
			A TrainingSharedConfig object requires an autoload of the type ai4u.MLPPPOAsyncSingleton to work.");
		}
		if (p != null)	
		{
			if (p is MLPPPOAsyncSingleton)
			{
				((MLPPPOAsyncSingleton)p).SharedConfig = this;
			}
			else if (p is ContinuousMLPPPOAsyncSingleton)
			{
				((ContinuousMLPPPOAsyncSingleton)p).SharedConfig = this;
			}
		}	 
	}
}
