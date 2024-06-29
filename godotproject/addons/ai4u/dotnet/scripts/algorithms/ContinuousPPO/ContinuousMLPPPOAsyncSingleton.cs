using Godot;
using System;
using System.IO;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using TorchSharp.Utils.tensorboard;
using TorchSharp;
using TorchSharp.Modules;
using System.Linq;
namespace ai4u;

public partial class ContinuousMLPPPOAsyncSingleton: Node
{
    private ConcurrentQueue< (string, string) > sampleCollections = new ConcurrentQueue< (string, string) >();

	public SummaryWriter summaryWriter;

    private ContinuousMLPPPO model;

    private PPOTrainingSharedConfig sharedConfig;
    
    public PPOTrainingSharedConfig SharedConfig
    {
        get
        {
            if (sharedConfig == null)
            {
                sharedConfig = new PPOTrainingSharedConfig();
                GetTree().Root.AddChild(sharedConfig);
            }
            return sharedConfig;
        }

        set
        {
            sharedConfig = value;
        }
    }

    private readonly object endEpisodeLock = new object();

    public ContinuousMLPPPO Model { 
            
        get 
        {
            return model;
        } 
    
        set 
        {
            model = value;
            if (model.algorithm == null)
            {
                var alg = new ContinuousMLPPPOAlgorithm();
                AddChild(alg);
                model.algorithm = alg;
            }
        }
    }



    private Dictionary<int, List<float> > rewards = new();

    private int numberOfWorkersUpdates = 0;
    private int numberOfGradientUpdates = 0;

    private int episodeCounter = 0;

    private int countModels = 0;

    private string logPath = "";
    private Task trainingLoop;

    private int finalizedWorkers = 0;

    private bool modelSaved = false;

    public void SetLogPath(string path)
    {
        this.logPath = path;
    }

    public void Put(string msg, string data)
    {
        sampleCollections.Enqueue( (msg, data) );
    }

    public override void _Ready()
    {
        summaryWriter = torch.utils.tensorboard.SummaryWriter(logPath, "ppoasync_log");
        trainingLoop = Task.Run(TrainingLoop);
        modelSaved = false;
    }



    public void AI4UNotificationClosed()
    {
        sampleCollections.Enqueue( ("done", "") );
        TrySaveModel();
    }


    private void TrySaveModel()
    {
        if (!modelSaved && Model != null)
        {
            modelSaved = true;
            Model.Save();
        }
    }

    public override void _ExitTree()
    {
        sampleCollections.Enqueue( ("done", "") );
        Task.WaitAll(trainingLoop);
        TrySaveModel();
    }

    public void TrainingLoop()
    {
        bool training = true;
        float criticLossAverage = 0;
        float policyLossAverage = 0;
        while (training)
        {

            if (sampleCollections.TryDequeue(out (string, string) item) && Model != null)
            { 
                if (item.Item1 ==  "done")
                {
                    training = false;
                    break;
                }
                else if (item.Item1 == "workdone")
                {
                    string[] losses = item.Item2.Split(" ");
                    criticLossAverage += float.Parse(losses[0]);
                    policyLossAverage += float.Parse(losses[1]);


                    numberOfWorkersUpdates++;
                    if ( (numberOfWorkersUpdates % SharedConfig.UpdateGradientInterval) == 0)
                    {
                        Model.algorithm.ApplyGradients(Model);
                        numberOfGradientUpdates++;
                        var criticLoss = criticLossAverage/SharedConfig.UpdateGradientInterval;
                        var policyLoss = policyLossAverage/SharedConfig.UpdateGradientInterval;
                        summaryWriter.add_scalar($"critic/loss", criticLoss,  numberOfGradientUpdates);
                        summaryWriter.add_scalar($"policy/loss", policyLoss, numberOfGradientUpdates);
                        GD.Print($"critic/loss: " + criticLoss);
                        GD.Print($"policy/Loss: " + policyLoss);
                        GD.Print($"Model Updates: {numberOfGradientUpdates}");
                        GD.Print($"Total of Workers Updates: {numberOfWorkersUpdates}");
                        criticLossAverage = 0;
                        policyLossAverage = 0;
                        
                    }
                }
                else if (item.Item1 == "endepisode")
                {
                    lock (endEpisodeLock)
                    {
                        episodeCounter += 1;
                        string[] args = item.Item2.Split(' ');
                        int e = int.Parse(args[0]);
                        float reward = float.Parse(args[1], System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
                        if (!rewards.ContainsKey(e))
                        {
                            rewards[e] = new List<float>();
                        }
                        var rewardsRef = rewards[e];
                        rewardsRef.Add(reward);
                        if (rewardsRef.Count >= Model.NumberOfEnvs)
                        {
                            var ar = rewardsRef.Average();
                            GD.Print("episode/reward " + ar);
                            summaryWriter.add_scalar("episode/reward", ar, e);
                            rewards.Remove(e);
                        }
                    }
                }
                else if (item.Item1 == "work finished")
                {
                    finalizedWorkers ++;
                    if (finalizedWorkers >= Model.NumberOfEnvs)
                    {
                        TrySaveModel();
                    }
                }
            }
        }
        TrySaveModel();
        GD.Print("Training loop was finalized!!!");
    }
}
