# Intro

This repo is building off of the code from https://github.com/yawgmoth/pyhanabi for my master's research on Human-AI Collaboration

## Our approach

Creating strategies to learn to collaborate with. If our agent can learn how to recognize the other player's strategy and then utilize structured knowledge of how to best coordinate with that strategy, we hypothesize that the Human-AI team can gain high rewards.


## Current strategy

Building the CHIEF agent which is intended to use a pool of human-like agents that reflect different types of players (differing in conventions/strategies). By using a probability distribution over each agent in the pool representing its likelihood of best representing our teammate, which we update with bayesian updates (the conditional probability update is the probability of the observed action by the teammate given each agent -- sample possible hands and get values for actions from agents to produce this probability distribution for updates). Finally, using the most up-to-date distribution over the agent pool, we can use that representation of the teammate to tailor our response to them for better total team-reward (for now we will just mirror the agents in the pool and if that is insufficient we will look into more advanced respones models)

# For questions
contact: arnavm@andrew.cmu.edu
