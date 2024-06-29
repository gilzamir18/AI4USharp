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

public class ContinuousMLPPPOMemory
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

public class ContinuousActorCritic : Module
{
    private readonly Sequential actorMean;
    private readonly Sequential actorLogStd;
    private readonly Sequential critic;

    public Sequential Actor => actorMean;
    public Sequential Critic => critic;

    public ContinuousActorCritic(int inputSize, int hiddenSize, int actionSize) : base(nameof(ActorCritic))
    {
        actorMean = Sequential(
            ("layer1", Linear(inputSize, hiddenSize)),
            ("relu1", ReLU()),
            ("layer2", Linear(hiddenSize, hiddenSize)),
            ("relu2", ReLU()),
            ("layer3", Linear(hiddenSize, actionSize))
        );

        actorLogStd = Sequential(
            ("layer1", Linear(inputSize, hiddenSize)),
            ("relu1", ReLU()),
            ("layer2", Linear(hiddenSize, hiddenSize)),
            ("relu2", ReLU()),
            ("layer3", Linear(hiddenSize, actionSize)),
            ("log_std", Tanh())  // Tanh para manter log_std em um intervalo razoável
        );

        critic = Sequential(
            ("layer1", Linear(inputSize, hiddenSize)),
            ("relu1", ReLU()),
            ("layer2", Linear(hiddenSize, hiddenSize)),
            ("relu2", ReLU()),
            ("layer3", Linear(hiddenSize, 1))
        );

        RegisterComponents();
    }

    public (Tensor mean, Tensor logStd) Act(Tensor input)
    {
        var mean = actorMean.forward(input);
        var logStd = actorLogStd.forward(input);
        return (mean, logStd);
    }

    public Tensor Evaluate(Tensor input)
    {
        return critic.forward(input);
    }

    public void Save(string prefix="model", string path="")
    {
        GD.Print(System.IO.Path.Join(path, prefix + "_critic.dat"));
        actorMean.save(System.IO.Path.Join(path, prefix + "_actor.dat"));
        critic.save(System.IO.Path.Join(path, prefix + "_critic.dat"));
    }

    public void Load(string prefix="model", string path="")
    {
        actorMean.load (System.IO.Path.Join(path, prefix + "_actor.dat"));
        critic.load(System.IO.Path.Join(path, prefix + "_critic.dat"));
    }
}


public partial class ContinuousMLPPPO: Node
{
    internal ContinuousActorCritic policy;
    internal ContinuousActorCritic oldPolicy;
    internal torch.optim.Optimizer optimizer;

    [Export]
    private Agent agent;

    [ExportGroup("Training Mode")]
    [Export]
    private bool trainingMode = true;
    [Export]
    internal ContinuousMLPPPOAlgorithm algorithm;
    [Export]
    internal bool shared = false;

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
                algorithm = new ContinuousMLPPPOAlgorithm();
                AddChild(algorithm);
            }
            this.inputSize = numInputs;
            this.hiddenSize = numHidden;
            this.outputSize = numOutputs;
        
            policy = new ContinuousActorCritic(inputSize, hiddenSize, outputSize);
            oldPolicy = new ContinuousActorCritic(inputSize, hiddenSize, outputSize);
            if (algorithm != null && optimizer == null)
            {
                GD.PrintRich("Warning: this model is in training mode!");
                optimizer = torch.optim.Adam(policy.parameters(), algorithm.LearningRate);
            }
            built = true;
            if (shared)
            {
                GetTree().Root.GetNode<ContinuousMLPPPOAsyncSingleton>("MLPPPOAsyncSingleton").Model = this;
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
        var (mean, logStd) = policy.Act(state);
        var std = logStd.exp();
        var action = mean + std * torch.randn_like(mean);
        return action;
    }

    public void Load(string prefix="model", string path="")
    {
        policy.Load(prefix, path);
        oldPolicy.Load(prefix, path);
    }
}
