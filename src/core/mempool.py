"""
Solana Mempool Implementation

The mempool manages pending transactions with:
- Fee-based prioritization for faster processing
- Account conflict detection for parallel execution
- Transaction validation and lifecycle management
- Gulf Stream-like leader forwarding simulation

This enables high-throughput transaction processing while maintaining consistency.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from heapq import heappush, heappop, heapify
from collections import defaultdict

from .transactions import SolanaTransaction
from .accounts import AccountRegistry


@dataclass
class MempoolEntry:
    """
    Entry in the mempool with priority and metadata.
    
    Uses negative priority for max-heap behavior (higher fees = higher priority).
    """
    priority: int                  # Negative fee for heap ordering
    timestamp: float              # When transaction was added
    transaction: SolanaTransaction
    retry_count: int = 0          # Number of execution attempts
    
    def __lt__(self, other: 'MempoolEntry') -> bool:
        """Priority comparison for heap ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp  # Older transactions first for ties


class SolanaMempool:
    """
    High-performance mempool for Solana transactions.
    
    Features:
    - Priority queue based on transaction fees
    - Account conflict detection for parallel batch creation
    - Automatic cleanup of invalid/expired transactions
    - Leader forwarding simulation (Gulf Stream)
    """
    
    def __init__(self, max_size: int = 10000, account_registry: Optional[AccountRegistry] = None):
        """
        Initialize mempool.
        
        Args:
            max_size: Maximum number of transactions to hold
            account_registry: Registry for account validation
        """
        self.max_size = max_size
        self.account_registry = account_registry
        
        # Priority queue for transaction ordering
        self._priority_queue: List[MempoolEntry] = []
        
        # Quick lookups
        self._transactions: Dict[str, MempoolEntry] = {}  # tx_hash -> entry
        self._account_usage: Dict[str, Set[str]] = defaultdict(set)  # account -> tx_hashes
        
        # Statistics
        self._stats = {
            'transactions_added': 0,
            'transactions_removed': 0,
            'transactions_rejected': 0,
            'account_conflicts_detected': 0,
            'total_fees_pending': 0,
        }
        
        # Configuration
        self.max_retry_count = 3
        self.transaction_ttl = 150.0  # 150 seconds (similar to Solana)
        
    def add_transaction(self, transaction: SolanaTransaction) -> bool:
        """
        Add transaction to mempool with validation.
        
        Returns:
            True if transaction was added, False if rejected
        """
        tx_hash = transaction.hash()
        
        # Check if already in mempool
        if tx_hash in self._transactions:
            return False
        
        # Validate transaction
        if not self._validate_transaction(transaction):
            self._stats['transactions_rejected'] += 1
            return False
        
        # Check for account conflicts
        if self._has_conflicts(transaction):
            self._stats['account_conflicts_detected'] += 1
            return False
        
        # Enforce size limit
        if len(self._transactions) >= self.max_size:
            if not self._evict_lowest_priority_transaction(transaction):
                self._stats['transactions_rejected'] += 1
                return False
        
        # Create mempool entry
        fee = transaction.calculate_fee()
        entry = MempoolEntry(
            priority=-fee,  # Negative for max-heap behavior
            timestamp=time.time(),
            transaction=transaction
        )
        
        # Add to data structures
        heappush(self._priority_queue, entry)
        self._transactions[tx_hash] = entry
        self._track_account_usage(transaction, tx_hash)
        
        # Update statistics
        self._stats['transactions_added'] += 1
        self._stats['total_fees_pending'] += fee
        
        return True
    
    def get_transactions_for_batch(self, max_count: int = 64, 
                                 max_compute_units: int = 48000000) -> List[SolanaTransaction]:
        """
        Get a batch of non-conflicting transactions for parallel execution.
        
        This implements Solana's scheduling algorithm for creating
        transaction batches that can execute simultaneously.
        
        Args:
            max_count: Maximum transactions in batch
            max_compute_units: Maximum compute budget for batch
            
        Returns:
            List of transactions that can execute in parallel
        """
        if not self._priority_queue:
            return []
        
        # Clean up expired transactions first
        self._cleanup_expired()
        
        selected_transactions = []
        used_accounts = set()
        total_compute_units = 0
        
        # Create a copy of priority queue for iteration
        temp_entries = list(self._priority_queue)
        heapify(temp_entries)
        
        while (temp_entries and 
               len(selected_transactions) < max_count and 
               total_compute_units < max_compute_units):
            
            entry = heappop(temp_entries)
            transaction = entry.transaction
            
            # Skip if transaction no longer in mempool
            if transaction.hash() not in self._transactions:
                continue
            
            # Check for account conflicts with already selected transactions
            tx_writable_accounts = transaction.get_writable_accounts()
            tx_readonly_accounts = transaction.get_readonly_accounts()
            
            # Conflict if any writable account is already used (read or write)
            if tx_writable_accounts & used_accounts:
                continue
            
            # Conflict if we want to read from an account that's being written
            if tx_readonly_accounts & used_accounts:
                continue
            
            # Check compute budget
            tx_compute_units = self._estimate_compute_units(transaction)
            if total_compute_units + tx_compute_units > max_compute_units:
                continue
            
            # Add to batch
            selected_transactions.append(transaction)
            used_accounts.update(tx_writable_accounts)
            total_compute_units += tx_compute_units
        
        # Remove selected transactions from mempool
        for transaction in selected_transactions:
            self._remove_transaction(transaction.hash())
        
        return selected_transactions
    
    def remove_transaction(self, tx_hash: str) -> bool:
        """Remove transaction from mempool by hash."""
        return self._remove_transaction(tx_hash)
    
    def get_transaction(self, tx_hash: str) -> Optional[SolanaTransaction]:
        """Get transaction by hash."""
        entry = self._transactions.get(tx_hash)
        return entry.transaction if entry else None
    
    def get_pending_count(self) -> int:
        """Get number of pending transactions."""
        return len(self._transactions)
    
    def get_total_pending_fees(self) -> int:
        """Get total fees of all pending transactions."""
        return self._stats['total_fees_pending']
    
    def get_statistics(self) -> Dict:
        """Get comprehensive mempool statistics."""
        current_stats = self._stats.copy()
        current_stats.update({
            'pending_transactions': len(self._transactions),
            'priority_queue_size': len(self._priority_queue),
            'account_conflicts_tracked': len(self._account_usage),
            'average_fee': (self._stats['total_fees_pending'] / len(self._transactions) 
                          if self._transactions else 0),
        })
        return current_stats
    
    def get_top_transactions(self, count: int = 10) -> List[SolanaTransaction]:
        """Get highest-fee transactions without removing them."""
        top_entries = []
        temp_queue = list(self._priority_queue)
        heapify(temp_queue)
        
        while temp_queue and len(top_entries) < count:
            entry = heappop(temp_queue)
            if entry.transaction.hash() in self._transactions:
                top_entries.append(entry.transaction)
        
        return top_entries
    
    def cleanup_invalid_transactions(self) -> int:
        """
        Remove invalid transactions from mempool.
        
        Returns:
            Number of transactions removed
        """
        if not self.account_registry:
            return 0
        
        invalid_hashes = []
        
        for tx_hash, entry in self._transactions.items():
            transaction = entry.transaction
            
            # Check if fee payer still has sufficient balance
            fee_payer = transaction.get_fee_payer()
            fee = transaction.calculate_fee()
            
            if self.account_registry.get_account_balance(fee_payer) < fee:
                invalid_hashes.append(tx_hash)
                continue
            
            # Check if all required accounts still exist
            for account_key in transaction.message.account_keys:
                if not self.account_registry.account_exists(account_key):
                    invalid_hashes.append(tx_hash)
                    break
        
        # Remove invalid transactions
        for tx_hash in invalid_hashes:
            self._remove_transaction(tx_hash)
        
        return len(invalid_hashes)
    
    def forward_to_leader(self, next_leaders: List[str]) -> Dict[str, List[SolanaTransaction]]:
        """
        Simulate Gulf Stream: forward transactions to upcoming leaders.
        
        This groups transactions by the leader most likely to process them.
        
        Returns:
            Dictionary mapping leader -> list of transactions
        """
        forwarded: Dict[str, List[SolanaTransaction]] = {leader: [] for leader in next_leaders}
        
        if not next_leaders:
            return forwarded
        
        # Simple round-robin forwarding (real Gulf Stream is more sophisticated)
        transactions = self.get_top_transactions(count=100)
        
        for i, transaction in enumerate(transactions):
            leader = next_leaders[i % len(next_leaders)]
            forwarded[leader].append(transaction)
        
        return forwarded
    
    # Private methods
    
    def _validate_transaction(self, transaction: SolanaTransaction) -> bool:
        """Validate transaction before adding to mempool."""
        try:
            # Basic signature verification
            if not transaction.verify_signatures():
                return False
            
            # Check transaction isn't too large
            if len(transaction.message.serialize()) > 1232:  # Solana's packet size limit
                return False
            
            # Verify fee payer has sufficient balance (if account registry available)
            if self.account_registry:
                fee_payer = transaction.get_fee_payer()
                fee = transaction.calculate_fee()
                
                if self.account_registry.get_account_balance(fee_payer) < fee:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _has_conflicts(self, transaction: SolanaTransaction) -> bool:
        """Check if transaction conflicts with existing mempool transactions."""
        tx_writable_accounts = transaction.get_writable_accounts()
        
        # Check if any writable account is already in use
        for account in tx_writable_accounts:
            if account in self._account_usage and self._account_usage[account]:
                return True
        
        return False
    
    def _track_account_usage(self, transaction: SolanaTransaction, tx_hash: str) -> None:
        """Track which accounts are used by this transaction."""
        for account_key in transaction.message.account_keys:
            self._account_usage[account_key].add(tx_hash)
    
    def _untrack_account_usage(self, transaction: SolanaTransaction, tx_hash: str) -> None:
        """Remove transaction from account usage tracking."""
        for account_key in transaction.message.account_keys:
            self._account_usage[account_key].discard(tx_hash)
            if not self._account_usage[account_key]:
                del self._account_usage[account_key]
    
    def _remove_transaction(self, tx_hash: str) -> bool:
        """Remove transaction from all data structures."""
        entry = self._transactions.get(tx_hash)
        if not entry:
            return False
        
        transaction = entry.transaction
        
        # Remove from mappings
        del self._transactions[tx_hash]
        self._untrack_account_usage(transaction, tx_hash)
        
        # Update statistics
        fee = transaction.calculate_fee()
        self._stats['transactions_removed'] += 1
        self._stats['total_fees_pending'] -= fee
        
        # Note: We don't remove from priority queue immediately for efficiency
        # Invalid entries will be skipped during iteration
        
        return True
    
    def _evict_lowest_priority_transaction(self, new_transaction: SolanaTransaction) -> bool:
        """
        Evict lowest priority transaction to make room for new one.
        
        Returns:
            True if eviction successful, False if new transaction has lower priority
        """
        if not self._transactions:
            return True
        
        # Find lowest priority transaction currently in mempool
        lowest_priority_entry = None
        lowest_priority_hash = None
        
        for tx_hash, entry in self._transactions.items():
            if (lowest_priority_entry is None or 
                entry.priority > lowest_priority_entry.priority):  # Remember: priority is negative
                lowest_priority_entry = entry
                lowest_priority_hash = tx_hash
        
        # Check if new transaction has higher priority
        new_fee = new_transaction.calculate_fee()
        if lowest_priority_entry and -lowest_priority_entry.priority >= new_fee:
            return False  # New transaction has lower or equal priority
        
        # Evict the lowest priority transaction
        if lowest_priority_hash:
            self._remove_transaction(lowest_priority_hash)
        
        return True
    
    def _cleanup_expired(self) -> int:
        """Remove expired transactions."""
        current_time = time.time()
        expired_hashes = []
        
        for tx_hash, entry in self._transactions.items():
            if current_time - entry.timestamp > self.transaction_ttl:
                expired_hashes.append(tx_hash)
        
        for tx_hash in expired_hashes:
            self._remove_transaction(tx_hash)
        
        return len(expired_hashes)
    
    def _estimate_compute_units(self, transaction: SolanaTransaction) -> int:
        """
        Estimate compute units for transaction.
        
        This is a simplified version of Solana's compute budget calculation.
        """
        base_units = 150000  # Base compute units per transaction
        
        # Add compute units based on instruction complexity
        for instruction in transaction.message.instructions:
            # Simple heuristic: 1000 CU per instruction + data size
            base_units += 1000 + len(instruction.data) * 10
        
        return min(base_units, 1400000)  # Cap at max CU per transaction
    
    def __len__(self) -> int:
        """Number of transactions in mempool."""
        return len(self._transactions)
    
    def __contains__(self, tx_hash: str) -> bool:
        """Check if transaction is in mempool."""
        return tx_hash in self._transactions
    
    def __iter__(self):
        """Iterate over transactions in priority order."""
        # Create sorted list of current transactions
        current_entries = [
            entry for entry in self._priority_queue 
            if entry.transaction.hash() in self._transactions
        ]
        current_entries.sort()  # Sort by priority
        
        for entry in current_entries:
            yield entry.transaction


class MempoolMonitor:
    """
    Monitor for tracking mempool performance and health.
    
    Provides insights into transaction processing patterns,
    fee market dynamics, and potential bottlenecks.
    """
    
    def __init__(self, mempool: SolanaMempool):
        self.mempool = mempool
        self.historical_stats = []
        self.monitoring_interval = 1.0  # seconds
        
    def take_snapshot(self) -> Dict:
        """Take a snapshot of current mempool state."""
        stats = self.mempool.get_statistics()
        stats['timestamp'] = time.time()
        
        # Add derived metrics
        if stats['pending_transactions'] > 0:
            fees = [tx.calculate_fee() for tx in self.mempool.get_top_transactions(100)]
            if fees:
                stats['median_fee'] = sorted(fees)[len(fees) // 2]
                stats['min_fee'] = min(fees)
                stats['max_fee'] = max(fees)
        
        self.historical_stats.append(stats)
        
        # Keep only recent history
        if len(self.historical_stats) > 3600:  # 1 hour at 1 second intervals
            self.historical_stats = self.historical_stats[-3600:]
        
        return stats
    
    def get_fee_market_analysis(self) -> Dict:
        """Analyze fee market trends."""
        if len(self.historical_stats) < 2:
            return {}
        
        recent_stats = self.historical_stats[-10:]  # Last 10 snapshots
        
        # Calculate trends
        pending_counts = [stat['pending_transactions'] for stat in recent_stats]
        avg_fees = [stat.get('average_fee', 0) for stat in recent_stats]
        
        return {
            'avg_pending_transactions': sum(pending_counts) / len(pending_counts),
            'pending_trend': 'increasing' if pending_counts[-1] > pending_counts[0] else 'decreasing',
            'avg_fee_trend': 'increasing' if avg_fees[-1] > avg_fees[0] else 'decreasing',
            'congestion_level': 'high' if pending_counts[-1] > 1000 else 'normal'
        }
    
    def print_status(self) -> None:
        """Print current mempool status."""
        stats = self.take_snapshot()
        
        print(f"\n{'='*50}")
        print("MEMPOOL STATUS")
        print(f"{'='*50}")
        print(f"Pending transactions: {stats['pending_transactions']:,}")
        print(f"Total pending fees: {stats['total_fees_pending']:,} lamports")
        print(f"Average fee: {stats['average_fee']:.0f} lamports")
        
        if 'median_fee' in stats:
            print(f"Median fee: {stats['median_fee']:.0f} lamports")
            print(f"Fee range: {stats['min_fee']:.0f} - {stats['max_fee']:.0f} lamports")
        
        print(f"Transactions added: {stats['transactions_added']:,}")
        print(f"Transactions removed: {stats['transactions_removed']:,}")
        print(f"Transactions rejected: {stats['transactions_rejected']:,}")
        print(f"Account conflicts detected: {stats['account_conflicts_detected']:,}")
        
        # Fee market analysis
        market_analysis = self.get_fee_market_analysis()
        if market_analysis:
            print(f"\nFee Market Analysis:")
            print(f"  Congestion level: {market_analysis['congestion_level']}")
            print(f"  Pending trend: {market_analysis['pending_trend']}")
            print(f"  Fee trend: {market_analysis['avg_fee_trend']}") 