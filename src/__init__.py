"""
Solana-like Blockchain Implementation in Python

A complete, educational implementation of Solana's revolutionary blockchain architecture
featuring Proof of History, account-based state, parallel execution, and more.

This implementation demonstrates the key innovations that make Solana the fastest
blockchain in the world, written with clarity and elegance in pure Python.

Key Features:
- ✅ Proof of History (PoH) for verifiable time ordering
- ✅ Account-based state model with rent mechanism  
- ✅ Instruction-based transactions for parallel execution
- ✅ System and SPL Token programs
- ✅ Proof of Stake consensus with leader election
- ✅ Tower BFT for Byzantine fault tolerance
- ✅ P2P networking with gossip protocol
- ✅ Sealevel parallel execution runtime
- ✅ Complete CLI interface

Author: Written in Peter Norvig's elegant style
Based on: Official Solana documentation and specifications
"""

__version__ = "1.0.0"
__author__ = "Solana from Scratch Educational Project"

# Core exports for easy importing
from .core import *
from .solana_cli import SolanaNode, SolanaCLI

__all__ = [
    # Core blockchain components
    'SolanaBlockchain',
    'SolanaAccount', 
    'AccountRegistry',
    'PoH',
    'PoHEntry',
    'PoHClock',
    'SolanaTransaction',
    'TransactionBuilder', 
    'Block',
    'SolanaMempool',
    'MempoolMonitor',
    
    # Main application classes
    'SolanaNode',
    'SolanaCLI',
] 