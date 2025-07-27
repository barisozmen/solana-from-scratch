"""
Solana Blockchain Core Implementation

This is the main blockchain class that orchestrates all components:
- Account registry for global state
- Proof of History for time ordering
- Block production and validation
- Transaction processing and mempool management
- Integration with consensus and networking layers

This represents the complete Solana ledger with all its innovative features.
"""

import time
import hashlib
from typing import List, Optional, Dict, Set, Tuple, Any
from dataclasses import dataclass

from .accounts import AccountRegistry, SolanaAccount
from .poh import PoH, PoHEntry, PoHClock
from .blocks import Block, BlockBuilder, BlockValidator, create_genesis_block
from .mempool import SolanaMempool, MempoolMonitor
from .transactions import SolanaTransaction, TransactionBuilder


@dataclass
class BlockchainStats:
    """Comprehensive blockchain statistics."""
    total_blocks: int
    total_transactions: int
    successful_transactions: int
    total_fees_collected: int
    total_compute_units_used: int
    current_slot: int
    total_accounts: int
    total_lamports: int
    average_tps: float
    current_leader: str


class SolanaBlockchain:
    """
    Complete Solana blockchain implementation.
    
    This is the central class that manages:
    - Global account state
    - Block production and validation
    - Transaction processing
    - Proof of History timing
    - Performance metrics
    """
    
    def __init__(self, genesis_leader: str = None, poh_iterations: int = 1000):
        """
        Initialize the blockchain.
        
        Args:
            genesis_leader: Leader for the genesis block
            poh_iterations: PoH iterations per entry (affects speed vs security)
        """
        # Core components
        self.accounts = AccountRegistry()
        self.poh = PoH(iterations_per_entry=poh_iterations)
        self.poh_clock = PoHClock(self.poh)
        self.mempool = SolanaMempool(account_registry=self.accounts)
        self.mempool_monitor = MempoolMonitor(self.mempool)
        
        # Configuration
        self.max_transactions_per_block = 64
        self.target_slot_time = 0.4  # 400ms slots like Solana
        self.recent_blockhash_limit = 150  # Keep last 150 block hashes
        
        # Blockchain state
        self.blocks: List[Block] = []
        self.block_hash_to_block: Dict[str, Block] = {}
        self.recent_blockhashes: List[str] = []
        
        # Statistics and metrics
        self._stats = {
            'total_transactions_processed': 0,
            'successful_transactions': 0,
            'total_fees_collected': 0,
            'total_compute_units_used': 0,
            'blocks_produced': 0,  # Will be 1 after genesis block
            'start_time': time.time(),
        }
        
        # Genesis setup (must be after configuration)
        self.genesis_leader = genesis_leader or self._generate_genesis_leader()
        self._create_genesis_block()
        
    def add_transaction(self, transaction: SolanaTransaction) -> bool:
        """
        Add transaction to mempool for processing.
        
        Returns:
            True if transaction was accepted, False if rejected
        """
        success = self.mempool.add_transaction(transaction)
        
        if success:
            print(f"Transaction {transaction.hash()[:8]}... added to mempool")
        else:
            print(f"Transaction {transaction.hash()[:8]}... rejected")
        
        return success
    
    def produce_block(self, leader: str = None) -> Optional[Block]:
        """
        Produce a new block as the current leader.
        
        This simulates Solana's block production process:
        1. Advance PoH
        2. Collect transactions from mempool
        3. Execute transactions
        4. Create block
        5. Update blockchain state
        
        Args:
            leader: Block leader (defaults to genesis leader)
            
        Returns:
            The produced block, or None if production failed
        """
        if leader is None:
            leader = self.genesis_leader
        
        try:
            # Get current state
            current_slot = self.current_slot() + 1
            parent_block = self.get_latest_block()
            
            # Create block builder
            builder = BlockBuilder(
                slot=current_slot,
                parent_block=parent_block,
                leader=leader
            )
            
            # Advance PoH and add entry
            poh_entry = self.poh.next_entry()
            builder.add_poh_entry(poh_entry)
            
            # Get transactions from mempool
            transactions = self.mempool.get_transactions_for_batch(
                max_count=self.max_transactions_per_block
            )
            
            print(f"Producing block {current_slot} with {len(transactions)} transactions")
            
            # Execute transactions and build block
            for transaction in transactions:
                success = self._execute_transaction(transaction)
                
                from .blocks import TransactionResult
                result = TransactionResult(
                    transaction=transaction,
                    success=success,
                    compute_units_consumed=transaction.calculate_fee() // 10,
                    fee_paid=transaction.calculate_fee() if success else 0
                )
                builder.add_transaction_result(result)
                
                # Update statistics
                self._stats['total_transactions_processed'] += 1
                if success:
                    self._stats['successful_transactions'] += 1
                    self._stats['total_fees_collected'] += transaction.calculate_fee()
            
            # Finalize block
            block = builder.finalize()
            
            # Validate block
            if not BlockValidator.validate_block(block, parent_block):
                print(f"Block {current_slot} validation failed")
                return None
            
            # Add to blockchain
            self._add_block(block)
            
            print(f"Block {current_slot} produced successfully")
            print(f"  Hash: {block.hash()[:16]}...")
            print(f"  Transactions: {len(block.transactions)}")
            print(f"  Fees collected: {block.total_fees_collected():,} lamports")
            
            return block
            
        except Exception as e:
            print(f"Block production failed: {e}")
            return None
    
    def get_account(self, pubkey: str) -> Optional[SolanaAccount]:
        """Get account by public key."""
        return self.accounts.get_account(pubkey)
    
    def create_account(self, pubkey: str, lamports: int = 0, 
                      space: int = 0, owner: str = None) -> SolanaAccount:
        """Create a new account."""
        return self.accounts.create_account(pubkey, lamports, space, owner)
    
    def get_balance(self, pubkey: str) -> int:
        """Get account balance in lamports."""
        return self.accounts.get_account_balance(pubkey)
    
    def transfer_lamports(self, from_pubkey: str, to_pubkey: str, amount: int) -> bool:
        """Transfer lamports between accounts."""
        return self.accounts.transfer_lamports(from_pubkey, to_pubkey, amount)
    
    def get_latest_block(self) -> Optional[Block]:
        """Get the most recent block."""
        return self.blocks[-1] if self.blocks else None
    
    def get_block_by_slot(self, slot: int) -> Optional[Block]:
        """Get block by slot number."""
        for block in self.blocks:
            if block.slot == slot:
                return block
        return None
    
    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        """Get block by hash."""
        return self.block_hash_to_block.get(block_hash)
    
    def current_slot(self) -> int:
        """Get current slot number."""
        latest_block = self.get_latest_block()
        return latest_block.slot if latest_block else 0
    
    def get_recent_blockhash(self) -> str:
        """Get a recent blockhash for transaction creation."""
        if self.recent_blockhashes:
            return self.recent_blockhashes[-1]
        return "1" * 64  # Fallback for empty blockchain
    
    def is_blockhash_valid(self, blockhash: str) -> bool:
        """Check if blockhash is recent enough for transactions."""
        return blockhash in self.recent_blockhashes
    
    def get_transaction(self, tx_hash: str) -> Optional[SolanaTransaction]:
        """Find transaction in blockchain or mempool."""
        # Check mempool first
        mempool_tx = self.mempool.get_transaction(tx_hash)
        if mempool_tx:
            return mempool_tx
        
        # Search blockchain
        for block in reversed(self.blocks):  # Start with recent blocks
            for result in block.transaction_results:
                if result.transaction_hash == tx_hash:
                    return result.transaction
        
        return None
    
    def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get detailed transaction status."""
        # Check mempool
        if tx_hash in self.mempool:
            return {
                'status': 'pending',
                'slot': None,
                'confirmations': 0,
                'error': None
            }
        
        # Search blockchain
        for block in reversed(self.blocks):
            result = block.get_transaction_result(tx_hash)
            if result:
                confirmations = self.current_slot() - block.slot
                return {
                    'status': 'confirmed' if result.success else 'failed',
                    'slot': block.slot,
                    'confirmations': confirmations,
                    'error': result.error_message
                }
        
        return {
            'status': 'not_found',
            'slot': None,
            'confirmations': 0,
            'error': 'Transaction not found'
        }
    
    def get_blockchain_stats(self) -> BlockchainStats:
        """Get comprehensive blockchain statistics."""
        elapsed_time = time.time() - self._stats['start_time']
        average_tps = self._stats['total_transactions_processed'] / elapsed_time if elapsed_time > 0 else 0
        
        return BlockchainStats(
            total_blocks=len(self.blocks),
            total_transactions=self._stats['total_transactions_processed'],
            successful_transactions=self._stats['successful_transactions'],
            total_fees_collected=self._stats['total_fees_collected'],
            total_compute_units_used=self._stats['total_compute_units_used'],
            current_slot=self.current_slot(),
            total_accounts=self.accounts.account_count(),
            total_lamports=self.accounts.total_lamports(),
            average_tps=average_tps,
            current_leader=self.genesis_leader
        )
    
    def validate_chain(self) -> bool:
        """Validate the entire blockchain."""
        if not self.blocks:
            return True
        
        # Validate genesis block
        if not BlockValidator.validate_block(self.blocks[0]):
            return False
        
        # Validate each subsequent block
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            parent_block = self.blocks[i - 1]
            
            if not BlockValidator.validate_block(current_block, parent_block):
                return False
        
        return True
    
    def display_status(self) -> None:
        """Display comprehensive blockchain status."""
        stats = self.get_blockchain_stats()
        
        print(f"\n{'='*60}")
        print("SOLANA BLOCKCHAIN STATUS")
        print(f"{'='*60}")
        
        # Blockchain metrics
        print(f"Current slot: {stats.current_slot}")
        print(f"Total blocks: {stats.total_blocks:,}")
        print(f"Total transactions: {stats.total_transactions:,}")
        print(f"Successful transactions: {stats.successful_transactions:,}")
        print(f"Success rate: {(stats.successful_transactions/stats.total_transactions*100):.1f}%" if stats.total_transactions > 0 else "N/A")
        print(f"Average TPS: {stats.average_tps:.2f}")
        
        # Economic metrics
        print(f"\nEconomic Metrics:")
        print(f"Total fees collected: {stats.total_fees_collected:,} lamports ({stats.total_fees_collected/1_000_000_000:.6f} SOL)")
        print(f"Total accounts: {stats.total_accounts:,}")
        print(f"Total lamports: {stats.total_lamports:,} ({stats.total_lamports/1_000_000_000:.6f} SOL)")
        
        # PoH metrics
        poh_info = self.poh.get_rate_info()
        print(f"\nProof of History:")
        print(f"Current PoH counter: {self.poh.counter}")
        print(f"PoH rate: {poh_info['entries_per_second']:.2f} entries/sec")
        print(f"Hash rate: {poh_info['iterations_per_second']:.0f} hashes/sec")
        
        # Mempool status
        print(f"\nMempool:")
        print(f"Pending transactions: {self.mempool.get_pending_count():,}")
        print(f"Pending fees: {self.mempool.get_total_pending_fees():,} lamports")
        
        # Recent blocks
        print(f"\nRecent Blocks:")
        recent_blocks = self.blocks[-5:] if len(self.blocks) > 5 else self.blocks
        for block in recent_blocks:
            print(f"  Slot {block.slot}: {len(block.transactions)} txs, {block.total_fees_collected():,} lamports fees")
    
    def create_transfer_transaction(self, from_keypair, to_pubkey: str, 
                                  amount: int) -> Optional[SolanaTransaction]:
        """
        Helper method to create a simple transfer transaction.
        
        Args:
            from_keypair: Tuple of (private_key, public_key)
            to_pubkey: Destination account public key
            amount: Amount in lamports
            
        Returns:
            Signed transaction ready for submission
        """
        from .transactions import create_transfer_instruction, sign_transaction
        
        private_key, from_pubkey = from_keypair
        
        # Ensure accounts exist
        if not self.accounts.account_exists(from_pubkey):
            print(f"Source account {from_pubkey[:8]}... does not exist")
            return None
        
        if not self.accounts.account_exists(to_pubkey):
            self.create_account(to_pubkey)
        
        # Build transaction
        recent_blockhash = self.get_recent_blockhash()
        
        builder = TransactionBuilder(from_pubkey, recent_blockhash)
        transfer_instruction = create_transfer_instruction(from_pubkey, to_pubkey, amount)
        builder.add_instruction(transfer_instruction)
        
        # Sign and return
        message = builder.build()
        return sign_transaction(message, [private_key])
    
    # Private methods
    
    def _generate_genesis_leader(self) -> str:
        """Generate a default genesis leader public key."""
        return hashlib.sha256(b"genesis_leader").hexdigest()
    
    def _create_genesis_block(self) -> None:
        """Create and add the genesis block."""
        initial_poh_hash = self.poh.get_current_entry().hash
        genesis_block = create_genesis_block(self.genesis_leader, initial_poh_hash)
        
        self._add_block(genesis_block)
        
        # Create initial system accounts
        self._create_system_accounts()
        
        print(f"Genesis block created")
        print(f"  Leader: {self.genesis_leader[:16]}...")
        print(f"  Hash: {genesis_block.hash()[:16]}...")
    
    def _create_system_accounts(self) -> None:
        """Create essential system accounts."""
        # System Program account
        system_account = SolanaAccount(
            lamports=1,
            data=b"system_program",
            owner="11111111111111111111111111111111",
            executable=True
        )
        self.accounts.set_account("11111111111111111111111111111111", system_account)
        
        # Genesis leader account with initial balance
        if not self.accounts.account_exists(self.genesis_leader):
            self.create_account(self.genesis_leader, lamports=1_000_000_000_000)  # 1000 SOL
    
    def _add_block(self, block: Block) -> None:
        """Add block to blockchain and update indexes."""
        self.blocks.append(block)
        self.block_hash_to_block[block.hash()] = block
        
        # Update recent blockhashes
        self.recent_blockhashes.append(block.hash())
        if len(self.recent_blockhashes) > self.recent_blockhash_limit:
            self.recent_blockhashes.pop(0)
        
        # Update statistics
        self._stats['blocks_produced'] += 1
    
    def _execute_transaction(self, transaction: SolanaTransaction) -> bool:
        """
        Execute a transaction and update account states.
        
        This is a simplified execution that handles basic system operations.
        A full implementation would include program invocation.
        """
        try:
            # Charge transaction fees
            fee = transaction.calculate_fee()
            fee_payer = transaction.get_fee_payer()
            
            fee_payer_account = self.accounts.get_account(fee_payer)
            if not fee_payer_account or fee_payer_account.lamports < fee:
                return False
            
            # Deduct fees
            updated_fee_payer = fee_payer_account.copy()
            updated_fee_payer.lamports -= fee
            self.accounts.set_account(fee_payer, updated_fee_payer)
            
            # Process instructions (simplified - only handle system transfers)
            for instruction in transaction.message.instructions:
                if not self._execute_instruction(instruction, transaction.message):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Transaction execution failed: {e}")
            return False
    
    def _execute_instruction(self, instruction, message) -> bool:
        """Execute a single instruction (simplified implementation)."""
        try:
            # Get program ID
            if instruction.program_id_index >= len(message.account_keys):
                return False
            
            program_id = message.account_keys[instruction.program_id_index]
            
            # Handle system program instructions
            if program_id == "11111111111111111111111111111111":  # System Program
                return self._execute_system_instruction(instruction, message)
            
            # For other programs, just return success (would invoke program in real implementation)
            return True
            
        except Exception:
            return False
    
    def _execute_system_instruction(self, instruction, message) -> bool:
        """Execute system program instruction."""
        if len(instruction.data) < 1:
            return False
        
        instruction_type = instruction.data[0]
        
        # Transfer instruction
        if instruction_type == 2 and len(instruction.accounts) >= 2:
            from_idx = instruction.accounts[0]
            to_idx = instruction.accounts[1]
            
            if (from_idx < len(message.account_keys) and 
                to_idx < len(message.account_keys) and 
                len(instruction.data) >= 9):
                
                from_account = message.account_keys[from_idx]
                to_account = message.account_keys[to_idx]
                amount = int.from_bytes(instruction.data[1:9], 'little')
                
                return self.accounts.transfer_lamports(from_account, to_account, amount)
        
        return True 