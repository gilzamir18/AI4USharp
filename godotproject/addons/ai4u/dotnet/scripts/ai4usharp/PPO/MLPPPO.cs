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


public class MLPPPO
{
    private readonly ActorCritic policy;
    private readonly ActorCritic oldPolicy;
    private readonly torch.optim.Optimizer optimizer;
    public float ClipParam {get; set;}
    public float Gamma {get; set;}
    public float Lambda {get; set;}
    [Export]
    public int Updates {get; set;} = 1;

    private int inputSize = 0;

    public MLPPPO(int inputSize, int hiddenSize, int actionSize, float clipParam, float gamma, float lambda, float learningRate)
    {
        policy = new ActorCritic(inputSize, hiddenSize, actionSize);
        oldPolicy = new ActorCritic(inputSize, hiddenSize, actionSize);
        optimizer = torch.optim.Adam(policy.parameters(), learningRate);
        this.ClipParam = clipParam;
        this.Gamma = gamma;
        this.Lambda = lambda;
        this.inputSize = inputSize;
    }

    public void Save()
    {
        policy.Save();
    }

    public  (float criticLoss, float policyLoss) Update(MLPPPOMemory memory)
    {
        var states = torch.stack(memory.states).detach();
        var actions = torch.stack(memory.actions).detach();
        var rewards = torch.tensor(memory.rewards.ToArray()).detach();
        var dones = torch.tensor(memory.dones.Select(d => d ? 1f : 0f).ToArray()).detach();

        var oldValues = oldPolicy.Evaluate(states.view(-1, inputSize)).squeeze().detach();
        if (oldValues.dim() == 0)
        {
            oldValues = oldValues.reshape(1);
        }

        var advantages = ComputeAdvantages(rewards, oldValues, dones).detach();
        var oldLogProbs = ComputeLogProbs(oldPolicy, states, actions).detach();
        float totalPolicyLoss = 0;
        float totalValueLoss = 0;
        for (int i = 0; i < Updates; i++)
        {
            var newLogProbs = ComputeLogProbs(policy, states, actions);
            var values = policy.Evaluate(states.view(-1, inputSize)).squeeze().detach();
            if (values.dim() == 0)
            {
                values = values.reshape(1);
            }
            var ratio = (newLogProbs - oldLogProbs).exp();
            var surr1 = ratio * advantages;
            var surr2 = ratio.clamp(1 - ClipParam, 1 + ClipParam) * advantages;

            var policyLoss = -torch.min(surr1, surr2).mean();
            var valueLoss = (values - rewards).pow(2).mean();
            totalPolicyLoss += policyLoss.item<float>();
            totalValueLoss += valueLoss.item<float>();
            optimizer.zero_grad();
            (policyLoss + valueLoss).backward();
            optimizer.step();
        }

        // Atualiza a política antiga
        oldPolicy.load_state_dict(policy.state_dict());
        return (totalValueLoss / Updates, totalPolicyLoss / Updates);
    }

    private Tensor ComputeAdvantages(Tensor rewards, Tensor values, Tensor dones)
    {
        var advantages = torch.zeros_like(rewards);
        var gae = torch.tensor(0.0f);
        var nextValue = torch.tensor(0.0f);
        
        for (long t = rewards.size(0) - 1; t >= 0; t--)
        {
            nextValue = t + 1 < rewards.size(0) ? values[t + 1] : torch.tensor(0.0f);
            var delta = rewards[t] + Gamma * nextValue * (1 - dones[t]) - values[t];
            gae = delta + Gamma * Lambda * (1 - dones[t]) * gae;
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

    public Tensor SelectAction(Tensor state)
    {
        var actionProbs = policy.Act(state);
        var action = torch.multinomial(actionProbs, 1);
        return action;
    }
}
