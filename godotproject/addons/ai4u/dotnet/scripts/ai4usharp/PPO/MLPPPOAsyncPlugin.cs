
using Godot;
using System;
using System.Threading;

namespace ai4u;

[Tool]
public partial class MLPPPOAsyncPlugin: Node
{
    public override void _Ready()
    {
        Thread t = new Thread(TrainingLoop);
    }

    public void TrainingLoop()
    {
        
    }
}