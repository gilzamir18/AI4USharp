using Godot;
using System;
using System.Runtime.CompilerServices;
using System.Text;

namespace ai4u
{
    /// <summary>
    /// MLPPPOSharedController.
    /// </summary>
    public partial class ContinuousMLPPPOSharedController: Controller
	{
		
        [Export]
        private string mainOutput = "move";

        [Export]
        private ContinuousMLPPPO model;


        private Controller controller;


		public override void OnSetup()
		{
            if (model == null)
            {
                GD.PrintErr(@"Error: field model is null. Set a MLPPPO node into this field.");
            }
            for (int i = 0; i < GetChildCount(); i++)
            {
                var child = GetChild(i);
                if (child is MLPPPOController && !model.IsInTrainingMode)
                {
                    controller = (ContinuousMLPPPOController) child;
                    ((ContinuousMLPPPOController) controller).model = model;
                    controller.agent = agent;
                    controller.OnSetup();
                    break;
                }
                else if (child is TrainController && model.IsInTrainingMode)
                {
                    controller = (TrainController) child;
                    controller.agent = agent;
                    controller.OnSetup();
                    break;
                }
            }
            if (controller == null)
            {
                GD.PrintErr(@"Error: MLPPOSharedController requires either a child of the type MLPPPOController 
                                or a child of the type TrainingController.");
            }
		}
		
		public override string GetAction()
		{
            if (GetStateAsString(0) == "envcontrol")
            {
                if (GetStateAsString(1).Contains("restart"))
                {
                    return ai4u.Utils.ParseAction("__restart__");
                }
                return ai4u.Utils.ParseAction("__noop__");			
            }
            if (controller != null)
            {
                return controller.GetAction();
            }
            else
            {
                return ai4u.Utils.ParseAction("__noop__");
            }
		}
		
		public override void NewStateEvent()
		{
            controller.GetStateFrom(this);
            if (GetStateAsString(0) != "envcontrol" && controller != null)
            {
			    controller.NewStateEvent();
            }
		}

		public override void OnReset(Agent agent)
		{
			controller.OnReset(agent);
		}
	}
}
