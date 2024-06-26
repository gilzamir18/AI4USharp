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

    public Sequential Actor => actor;
    public Sequential Critic => critic;

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


public partial class MLPPPO: Node
{
    internal ActorCritic policy;
    internal ActorCritic oldPolicy;
    internal torch.optim.Optimizer optimizer;

    [Export]
    private Agent agent;

    [ExportGroup("Training Mode")]
    [Export]
    private bool trainingMode = true;
    [Export]
    internal MLPPPOAlgorithm algorithm;

    public int NumberOfEnvs {get; set;} = 0;

    private int inputSize = 2;
    private int outputSize = 4;
    private int hiddenSize = 32;

    public int InputSize => inputSize;

    private bool built = false;

    public bool IsInTrainingMode => trainingMode;

    internal void Build(int numInputs, int numHidden, int numOutputs)
    {
        if (!built)
        {
            if (trainingMode)
            {
                algorithm = new MLPPPOAlgorithm();
                AddChild(algorithm);
            }
            this.inputSize = numInputs;
            this.hiddenSize = numHidden;
            this.outputSize = numOutputs;
        
            policy = new ActorCritic(inputSize, hiddenSize, outputSize);
            oldPolicy = new ActorCritic(inputSize, hiddenSize, outputSize);
            if (algorithm != null && optimizer == null)
            {
                GD.PrintRich("Warning: this model is in training mode!");
                optimizer = torch.optim.Adam(policy.parameters(), algorithm.LearningRate);
            }
            built = true;
            if (trainingMode)
            {
                GetTree().Root.GetNode<MLPPPOAsyncSingleton>("MLPPPOAsyncSingleton").Model = this;
            }
        }
    }

    public void Save()
    {
        policy.Save();
    }

    public void SyncWith(MLPPPO model)
    {
        policy.Actor.load_state_dict(model.policy.Actor.state_dict());
        policy.Critic.load_state_dict(model.policy.Critic.state_dict());
        oldPolicy.Actor.load_state_dict(model.oldPolicy.Actor.state_dict());
        oldPolicy.Critic.load_state_dict(model.oldPolicy.Critic.state_dict());
    }

    public Tensor SelectAction(Tensor state)
    {
        var actionProbs = policy.Act(state);
        var action = torch.multinomial(actionProbs, 1);
        return action;
    }

    public void Load(string prefix="model", string path="")
    {
        policy.Load(prefix, path);
        oldPolicy.Load(prefix, path);
    }
}
