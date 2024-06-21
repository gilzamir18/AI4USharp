using System.Collections;
using System.Collections.Generic;
using Godot;

namespace ai4u
{
	public partial class RewardSensor : Sensor
	{
		public float rewardScale = 1.0f;
		public override void OnSetup(Agent agent)
		{
			this.agent = agent;
			perceptionKey = "reward";
			type = SensorType.sfloat;
			shape = new int[]{stackedObservations};
		}

		public override float GetFloatValue()
		{
			return agent.Reward * rewardScale;
		}
	}
}
