# AI4USharp - Artificial Inteligence For You

AI4USharp tool is an agent framework for modeling virtual reality and game environment. This repo we keep the reference implementation for Godot Game Engine.

The Godot version of the AI4U get advantage of the Godot architecture and facilitate agent specification by means of an agent abstraction. In Godot, AI4U provides a alternative approach to modeling Non-Player Characters (NPC). Although, developers can apply this tool in others situations, for example, the modeling environment for artificial intelligence experiments.

Agent abstraction defines an agent living in a environment and interacting with this environment by means of sensors and actuators. So, NPC specification is a kind of agent specification. Agent's components are: sensors, actuators, events, reward functions and brain. Sensors and actuators are the interface between agents and environments. A sensor provides data to an agent's brain, while actuators send actions from agent to environment. A brain is a script that proccessing sensors' data e made a decision (selects an action by time).

# Use AI4USharp
AI4USharp is the implementation of reinforcement learning  using TorchSharp. The current implementation is in an experimental state and contains only a simple implementation of the algorithm PPO.


# Requirements
* Godot 4.2.2 Mono Version.
* TorchSharp-cpu (version >= 0.102).
* Tested in Windows 11 or Ubuntu 24.04.

The minimum recommended hardware for AI4U is a computer with at least a GeForce 1050ti (4GB VRAM), 8GB of RAM, and at least 20GB of SSD storage. Naturally, the memory requirement can increase significantly if complex inputs are used in the agent's sensors (such as images) and if algorithms like *Soft-Actor-Critic* (SAC) and DQN are employed. For truly interesting use cases, such as using SAC with an image sensor, a computer with at least 24GB of RAM and a high-end GPU is necessary. For games, we recommend modest sensor configurations, such as moderate use of RayCasting.

# Maintainers
AI4U is currently maintained by Gilzamir Gomes (gilzamir_gomes@uvanet.br).

Important Note: We do not do technical support, nor consulting and don't answer personal questions per email.
How To Contribute
To any interested in making the AI4USharp better, there is still some documentation that needs to be done. If you want to contribute, please read CONTRIBUTING.md guide first.

