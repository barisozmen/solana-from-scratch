"""
Solana Block Structure Implementation

Blocks in Solana contain:
- Proof of History entries for verifiable time ordering
- Transactions with their execution results
- Block metadata including leader information
- Cryptographic links to previous blocks

This implements a simplified but accurate representation of Solana's block format.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .poh import PoHEntry
from .transactions import SolanaTransaction


@dataclass
class BlockHeader:
    """
    Block header containing essential metadata.
    
    This is kept separate from the block body for efficient
    header-only synchronization and validation.
    """
    slot: int                      # Slot number (Solana's time unit)
    parent_slot: int              # Previous block's slot
    previous_blockhash: str       # Hash of previous block
    poh_hash: str                 # Proof of History hash for this slot
    transaction_count: int        # Number of transactions in block
    leader: str                   # Public key of block leader
    timestamp: float              # Wall-clock time when block was created
    
    def hash(self) -> str:
        """Compute header hash for linking and identification."""
        header_string = (
            f"{self.slot}{self.parent_slot}{self.previous_blockhash}"
            f"{self.poh_hash}{self.transaction_count}{self.leader}{self.timestamp}"
        ).encode()
        return hashlib.sha256(header_string).hexdigest()


@dataclass
class TransactionResult:
    """
    Result of transaction execution within a block.
    
    Tracks both the transaction and its execution outcome,
    including any errors or state changes.
    """
    transaction: SolanaTransaction
    success: bool
    error_message: Optional[str] = None
    compute_units_consumed: int = 0
    fee_paid: int = 0
    logs: List[str] = field(default_factory=list)
    
    @property
    def transaction_hash(self) -> str:
        return self.transaction.hash()


@dataclass
class Block:
    """
    Complete Solana block with header, PoH entries, and transactions.
    
    This represents a finalized block that has been accepted by the network
    and contains all the information needed for validation and state updates.
    """
    header: BlockHeader
    poh_entries: List[PoHEntry]
    transaction_results: List[TransactionResult]
    
    def __post_init__(self):
        """Validate block consistency after creation."""
        if self.header.transaction_count != len(self.transaction_results):
            raise ValueError("Transaction count mismatch in block header")
    
    def hash(self) -> str:
        """
        Compute block hash from header and content.
        
        This creates a unique identifier for the block that depends
        on all its content, enabling tamper detection.
        """
        # Start with header hash
        content_parts = [self.header.hash()]
        
        # Add PoH entries
        poh_hashes = [entry.hash for entry in self.poh_entries]
        content_parts.append("".join(poh_hashes))
        
        # Add transaction hashes
        tx_hashes = [result.transaction_hash for result in self.transaction_results]
        content_parts.append("".join(tx_hashes))
        
        # Compute final hash
        block_string = "".join(content_parts).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @property
    def slot(self) -> int:
        """Get block slot number."""
        return self.header.slot
    
    @property
    def transactions(self) -> List[SolanaTransaction]:
        """Get all transactions in this block."""
        return [result.transaction for result in self.transaction_results]
    
    @property
    def successful_transactions(self) -> List[SolanaTransaction]:
        """Get only successfully executed transactions."""
        return [result.transaction for result in self.transaction_results if result.success]
    
    @property
    def failed_transactions(self) -> List[SolanaTransaction]:
        """Get failed transactions."""
        return [result.transaction for result in self.transaction_results if not result.success]
    
    def get_transaction_result(self, tx_hash: str) -> Optional[TransactionResult]:
        """Get execution result for a specific transaction."""
        for result in self.transaction_results:
            if result.transaction_hash == tx_hash:
                return result
        return None
    
    def total_fees_collected(self) -> int:
        """Calculate total fees collected in this block."""
        return sum(result.fee_paid for result in self.transaction_results)
    
    def total_compute_units_used(self) -> int:
        """Calculate total compute units consumed."""
        return sum(result.compute_units_consumed for result in self.transaction_results)
    
    def success_rate(self) -> float:
        """Calculate transaction success rate."""
        if not self.transaction_results:
            return 1.0
        successful = sum(1 for result in self.transaction_results if result.success)
        return successful / len(self.transaction_results)
    
    def summary(self) -> Dict[str, Any]:
        """Get block summary statistics."""
        return {
            "slot": self.slot,
            "hash": self.hash()[:16] + "...",
            "leader": self.header.leader[:16] + "...",
            "transaction_count": len(self.transaction_results),
            "successful_transactions": len(self.successful_transactions),
            "failed_transactions": len(self.failed_transactions),
            "success_rate": f"{self.success_rate():.2%}",
            "total_fees": self.total_fees_collected(),
            "compute_units_used": self.total_compute_units_used(),
            "poh_entries": len(self.poh_entries),
            "timestamp": self.header.timestamp
        }


class BlockBuilder:
    """
    Builder for constructing blocks with proper validation.
    
    This handles the complex process of creating valid blocks including:
    - PoH entry generation
    - Transaction execution
    - Fee collection
    - Block finalization
    """
    
    def __init__(self, slot: int, parent_block: Optional[Block], leader: str):
        """
        Initialize block builder.
        
        Args:
            slot: Slot number for this block
            parent_block: Previous block (None for genesis)
            leader: Public key of block leader
        """
        self.slot = slot
        self.parent_block = parent_block
        self.leader = leader
        self.poh_entries: List[PoHEntry] = []
        self.transaction_results: List[TransactionResult] = []
        
        # Derived values
        self.parent_slot = parent_block.slot if parent_block else 0
        self.previous_blockhash = parent_block.hash() if parent_block else "0" * 64
        self.timestamp = time.time()
    
    def add_poh_entry(self, entry: PoHEntry) -> 'BlockBuilder':
        """Add a PoH entry to the block."""
        self.poh_entries.append(entry)
        return self
    
    def add_transaction_result(self, result: TransactionResult) -> 'BlockBuilder':
        """Add a transaction execution result."""
        self.transaction_results.append(result)
        return self
    
    def execute_transaction(self, transaction: SolanaTransaction, 
                          runtime: 'SolanaRuntime') -> 'BlockBuilder':
        """
        Execute a transaction and add the result.
        
        This simulates the actual execution process including
        fee deduction and instruction processing.
        """
        try:
            # Calculate and charge fees
            fee = transaction.calculate_fee()
            fee_payer = transaction.get_fee_payer()
            
            # Check if fee payer can afford the transaction
            fee_payer_account = runtime.get_account(fee_payer)
            if not fee_payer_account or fee_payer_account.lamports < fee:
                result = TransactionResult(
                    transaction=transaction,
                    success=False,
                    error_message="Insufficient funds for fees",
                    fee_paid=0
                )
                self.add_transaction_result(result)
                return self
            
            # Execute instructions
            execution_success = runtime.execute_transaction(transaction)
            
            # Create result
            result = TransactionResult(
                transaction=transaction,
                success=execution_success,
                compute_units_consumed=transaction.calculate_fee() // 10,  # Simplified
                fee_paid=fee if execution_success else 0,
                logs=[]  # Would contain program logs in real implementation
            )
            
            self.add_transaction_result(result)
            
        except Exception as e:
            # Handle unexpected execution errors
            result = TransactionResult(
                transaction=transaction,
                success=False,
                error_message=str(e),
                fee_paid=0
            )
            self.add_transaction_result(result)
        
        return self
    
    def finalize(self) -> Block:
        """
        Finalize and create the complete block.
        
        This performs final validation and creates the immutable block.
        """
        # Ensure we have at least one PoH entry
        if not self.poh_entries:
            raise ValueError("Block must contain at least one PoH entry")
        
        # Use the latest PoH entry for the block hash
        latest_poh = self.poh_entries[-1]
        
        # Create header
        header = BlockHeader(
            slot=self.slot,
            parent_slot=self.parent_slot,
            previous_blockhash=self.previous_blockhash,
            poh_hash=latest_poh.hash,
            transaction_count=len(self.transaction_results),
            leader=self.leader,
            timestamp=self.timestamp
        )
        
        # Create and validate block
        block = Block(
            header=header,
            poh_entries=self.poh_entries.copy(),
            transaction_results=self.transaction_results.copy()
        )
        
        return block


class BlockValidator:
    """
    Validates blocks according to Solana's consensus rules.
    
    This ensures blocks are properly formed and contain valid
    PoH sequences and transaction executions.
    """
    
    @staticmethod
    def validate_block(block: Block, parent_block: Optional[Block] = None) -> bool:
        """
        Validate a complete block.
        
        Checks:
        - Block structure integrity
        - PoH entry validity
        - Transaction execution results
        - Cryptographic links
        """
        try:
            # Basic structure validation
            if not BlockValidator._validate_structure(block):
                return False
            
            # Parent block linkage
            if parent_block and not BlockValidator._validate_parent_link(block, parent_block):
                return False
            
            # PoH sequence validation
            if not BlockValidator._validate_poh_sequence(block):
                return False
            
            # Transaction validation
            if not BlockValidator._validate_transactions(block):
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def _validate_structure(block: Block) -> bool:
        """Validate basic block structure."""
        # Check header consistency
        if block.header.transaction_count != len(block.transaction_results):
            return False
        
        # Check PoH entries exist
        if not block.poh_entries:
            return False
        
        # Check header hash matches content
        if block.header.poh_hash != block.poh_entries[-1].hash:
            return False
        
        return True
    
    @staticmethod
    def _validate_parent_link(block: Block, parent_block: Block) -> bool:
        """Validate link to parent block."""
        # Check slot sequence
        if block.header.slot <= parent_block.header.slot:
            return False
        
        # Check parent hash reference
        if block.header.previous_blockhash != parent_block.hash():
            return False
        
        # Check parent slot reference
        if block.header.parent_slot != parent_block.header.slot:
            return False
        
        return True
    
    @staticmethod
    def _validate_poh_sequence(block: Block) -> bool:
        """Validate PoH entry sequence."""
        # Check each PoH entry is properly formed
        for entry in block.poh_entries:
            if entry.counter <= 0 or not entry.hash:
                return False
        
        # Check sequence ordering
        for i in range(1, len(block.poh_entries)):
            current = block.poh_entries[i]
            previous = block.poh_entries[i - 1]
            
            if current.counter <= previous.counter:
                return False
        
        return True
    
    @staticmethod
    def _validate_transactions(block: Block) -> bool:
        """Validate transaction execution results."""
        for result in block.transaction_results:
            # Verify transaction signature
            if not result.transaction.verify_signatures():
                return False
            
            # Check fee calculation
            expected_fee = result.transaction.calculate_fee()
            if result.success and result.fee_paid != expected_fee:
                return False
        
        return True


def create_genesis_block(leader: str, initial_poh_hash: str) -> Block:
    """
    Create the genesis block for the blockchain.
    
    This is the first block that bootstraps the entire blockchain.
    """
    from .poh import PoHEntry
    
    # Create genesis PoH entry
    genesis_poh = PoHEntry(counter=1, hash=initial_poh_hash)
    
    # Create genesis header
    header = BlockHeader(
        slot=0,
        parent_slot=0,
        previous_blockhash="0" * 64,
        poh_hash=genesis_poh.hash,
        transaction_count=0,
        leader=leader,
        timestamp=time.time()
    )
    
    # Create genesis block
    return Block(
        header=header,
        poh_entries=[genesis_poh],
        transaction_results=[]
    ) 