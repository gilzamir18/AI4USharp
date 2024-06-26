using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using System.Linq;
using Godot;

namespace ai4u;

public partial class MLPPPOAlgorithm: Node
{

    [Export]
    private int updates = 1;

    [Export]
    private float learningRate = 0.00025f;

    [Export]
    private float gamma = 0.99f;
    
    [Export]
    private float lambda = 0.95f;
    
    [Export]
    private float clipParam = 0.2f; 
    
    public float LearningRate => learningRate;
    private readonly object updateLock = new object();

    public  (float criticLoss, float policyLoss) Update(MLPPPO  model, MLPPPOMemory memory, bool accumulate = false)
    {
        lock(updateLock)
        {
            var states = torch.stack(memory.states).detach();
            var actions = torch.stack(memory.actions).detach();
            var rewards = torch.tensor(memory.rewards.ToArray()).detach();
            var dones = torch.tensor(memory.dones.Select(d => d ? 1f : 0f).ToArray()).detach();

            var oldValues = model.oldPolicy.Evaluate(states.view(-1, model.InputSize)).squeeze().detach();
            if (oldValues.dim() == 0)
            {
                oldValues = oldValues.reshape(1);
            }

            var advantages = ComputeAdvantages(rewards, oldValues, dones).detach();
            var oldLogProbs = ComputeLogProbs(model.oldPolicy, states, actions).detach();
            float totalPolicyLoss = 0;
            float totalValueLoss = 0;
            for (int i = 0; i < updates; i++)
            {
                var newLogProbs = ComputeLogProbs(model.policy, states, actions);
                var values = model.policy.Evaluate(states.view(-1, model.InputSize)).squeeze().detach();
                if (values.dim() == 0)
                {
                    values = values.reshape(1);
                }
                var ratio = (newLogProbs - oldLogProbs).exp();
                var surr1 = ratio * advantages;
                var surr2 = ratio.clamp(1 - clipParam, 1 + clipParam) * advantages;

                var policyLoss = -torch.min(surr1, surr2).mean();
                var valueLoss = (values - rewards).pow(2).mean();
                totalPolicyLoss += policyLoss.item<float>();
                totalValueLoss += valueLoss.item<float>();
                if (!accumulate)
                {
                    model.optimizer.zero_grad();
                }
                (policyLoss + valueLoss).backward();
                if (!accumulate)
                {
                    model.optimizer.step();
                }
            }
        

            if (!accumulate)
            {
                // Update the old policy
                model.oldPolicy.load_state_dict(model.policy.state_dict());
            }
            return (totalValueLoss / updates, totalPolicyLoss / updates);
        }
    }

    public void ApplyGradients(MLPPPO  model)
    {
        lock (updateLock)
        {
            model.optimizer.step();
            model.optimizer.zero_grad();
            model.oldPolicy.load_state_dict(model.policy.state_dict());
        }
    }

    private Tensor ComputeAdvantages(Tensor rewards, Tensor values, Tensor dones)
    {
        var advantages = torch.zeros_like(rewards);
        var gae = torch.tensor(0.0f);
        var nextValue = torch.tensor(0.0f);
        
        for (long t = rewards.size(0) - 1; t >= 0; t--)
        {
            nextValue = t + 1 < rewards.size(0) ? values[t + 1] : torch.tensor(0.0f);
            var delta = rewards[t] + gamma * nextValue * (1 - dones[t]) - values[t];
            gae = delta + gamma * lambda * (1 - dones[t]) * gae;
            advantages[t] = gae;
        }
        return advantages;
    }

    private Tensor ComputeLogProbs(ActorCritic policy, Tensor states, Tensor actions)
    {
        var actionProbs = policy.Act(states);
        actions = actions.view(-1, 1);
        var logProbs = torch.log(actionProbs.gather(1, actions));
        return logProbs.squeeze();
    }
}