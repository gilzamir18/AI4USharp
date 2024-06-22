using Godot;
using System;
using System.IO;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using TorchSharp.Utils.tensorboard;
using TorchSharp;
using TorchSharp.Modules;
namespace ai4u;

public partial class MLPPPOAsyncSingleton: Node
{
    private ConcurrentQueue< (string, MLPPPOMemory, MLPPPO) > sampleCollections = new ConcurrentQueue< (string, MLPPPOMemory, MLPPPO) >();

	public SummaryWriter summaryWriter;

    private MLPPPO model;

    public MLPPPO Model { 
            
        get 
        {
            return model;
        } 
    
        set 
        {
            model = value;
            if (model.algorithm == null)
            {
                var alg = new MLPPPOAlgorithm();
                AddChild(alg);
                model.algorithm = alg;
            }
        }
    }

    private Dictionary<string, string> properties = new Dictionary<string, string>();

    private int globalStep = 0;

    private int countModels = 0;

    private string logPath = "";
    private Task trainingLoop;

    public void PutSample(string msg, MLPPPO model, MLPPPOMemory sample)
    {
        sampleCollections.Enqueue( (msg, sample, model) );
    }

    public override void _Ready()
    {
        summaryWriter = torch.utils.tensorboard.SummaryWriter(logPath, "loss_log");
        trainingLoop = Task.Run(TrainingLoop);
    }


    public override void _Notification(int what)
    {
        if (what == NotificationWMCloseRequest)
        {
            sampleCollections.Enqueue( ("done", null, null) );
            Task.WaitAll(trainingLoop);
            Model.Save();
        }
    }

    public override void _ExitTree()
    {
        sampleCollections.Enqueue( ("done", null, null) );
        Task.WaitAll(trainingLoop);
    }

    public void TrainingLoop()
    {
        bool training = true;
        while (training)
        {

            if (sampleCollections.TryDequeue(out (string, MLPPPOMemory, MLPPPO) item) && Model != null)
            { 
                if (item.Item1 ==  "done")
                {
                    training = false;
                    break;
                }
                else
                {
                    Model.oldPolicy.load_state_dict(item.Item3.policy.state_dict());
                    var (criticLoss, policyLoss) = Model.algorithm.Update(Model, item.Item2);
                    item.Item3.policy.load_state_dict(Model.policy.state_dict());
                    item.Item3.oldPolicy.load_state_dict(Model.oldPolicy.state_dict());

                    summaryWriter.add_scalar("critic/loss", criticLoss, globalStep);
                    summaryWriter.add_scalar("policy/Loss", policyLoss, globalStep);
                    //item.Clear();
                    if (globalStep % 100 == 0 && globalStep > 0)
                    {
                        GD.Print("critic/loss: " + criticLoss);
                        GD.Print("policy/Loss: " + policyLoss);
                    }
                    globalStep++;
                }
            }
        }
        Model.Save();
        GD.Print("Training loop was finalized!!!");
    }

    public void SyncModel(MLPPPO model)
    {
        model.SyncWith(Model);
    }
}
