"""
Solana Consensus Implementation

This module implements Solana's consensus mechanism consisting of:
- Proof of Stake (PoS): Validator staking and leader selection
- Tower BFT: Byzantine fault-tolerant consensus with PoH integration
- Fork choice: Heaviest chain selection based on stake-weighted votes
- Validator network: Validator management and vote processing

This demonstrates how Solana achieves fast finality and high throughput
while maintaining security through cryptographic proofs and economic incentives.
"""

from .validators import Validator, ValidatorSet, StakeRegistry
from .tower_bft import TowerBFT, Vote, VoteState, ForkChoice
from .leader_schedule import LeaderSchedule, SlotLeaderProvider
from .consensus_engine import ConsensusEngine

__all__ = [
    'Validator',
    'ValidatorSet', 
    'StakeRegistry',
    'TowerBFT',
    'Vote',
    'VoteState',
    'ForkChoice',
    'LeaderSchedule',
    'SlotLeaderProvider',
    'ConsensusEngine'
] 