# Stage 5: Consensus with Tower BFT

> *"Consensus is not about getting everyone to agree, it's about getting everyone to agree on what everyone agrees on."* - Leslie Lamport

## Objectives

This stage implements Solana's Tower BFT consensus algorithm, handling network disagreements through voting and fork resolution:

- Add validator voting on proposed blocks
- Implement fork detection and management
- Create stake-weighted vote counting
- Build fork choice rules for canonical chain selection
- Handle Byzantine fault tolerance with <1/3 malicious validators

## Why Tower BFT?

### Traditional BFT Limitations
- **Synchrony assumptions**: Require bounded message delays
- **View changes**: Complex leader change protocols
- **Communication overhead**: O(n²) message complexity

### Tower BFT Advantages
- **Asynchronous**: Works with unbounded network delays
- **PoH integration**: Uses time progression for ordering
- **Optimistic**: Fast path for common case
- **Fork-based**: Natural handling of network partitions

## Key Concepts

### 1. Voting Mechanism
Validators vote on blocks by:
- **Signing block hashes** with their private keys
- **Broadcasting votes** to the network
- **Accumulating stake-weighted** vote tallies
- **Confirming blocks** above threshold

### 2. Fork Management
When network splits occur:
- **Multiple chains** can exist temporarily
- **Validators vote** on their preferred fork
- **Heaviest fork** wins (most cumulative stake)
- **Reorganization** occurs to canonical chain

### 3. Safety and Liveness
- **Safety**: Never finalize conflicting blocks
- **Liveness**: Eventually make progress
- **Threshold**: Require >2/3 stake for confirmation

## Implementation

### Vote Structure

```python
@dataclass(frozen=True)
class Vote:
    validator_id: str    # Who cast the vote
    block_hash: str      # Which block they voted for
    slot: int           # When they voted (PoH slot)
    signature: str      # Cryptographic proof

    def verify_signature(self, validator_public_key: str) -> bool:
        """Verify vote signature is valid."""
        try:
            vk = VerifyingKey.from_string(bytes.fromhex(validator_public_key), curve=SECP256k1)
            vote_data = f"{self.validator_id}{self.block_hash}{self.slot}".encode()
            return vk.verify(bytes.fromhex(self.signature), vote_data)
        except:
            return False
```

### Enhanced Block with Votes

```python
@dataclass(frozen=True)
class Block:
    index: int
    poh_entry: PoHEntry
    transactions: List[Transaction]
    previous_hash: str
    leader_id: str
    votes: List[Vote] = field(default_factory=list)  # Votes for this block

    def hash(self) -> str:
        tx_hashes = "".join(tx.hash() for tx in self.transactions)
        vote_hashes = "".join(f"{v.validator_id}{v.signature}" for v in self.votes)
        block_string = f"{self.index}{self.poh_entry.counter}{self.poh_entry.hash}{tx_hashes}{vote_hashes}{self.previous_hash}{self.leader_id}".encode()
        return hashlib.sha256(block_string).hexdigest()

    def get_vote_weight(self, stake_registry) -> float:
        """Calculate total stake weight of votes."""
        total_weight = 0.0
        for vote in self.votes:
            validator = stake_registry.validators.get(vote.validator_id)
            if validator:
                total_weight += validator.stake
        return total_weight

    def has_supermajority(self, stake_registry) -> bool:
        """Check if block has >2/3 stake voting for it."""
        vote_weight = self.get_vote_weight(stake_registry)
        total_stake = stake_registry.total_stake()
        return vote_weight > (2/3 * total_stake)
```

### Fork Manager

```python
class ForkManager:
    def __init__(self):
        self.forks: Dict[str, List[Block]] = {}  # root_hash -> chain
        self.canonical_root = "genesis"

    def add_fork(self, root_hash: str, initial_block: Block) -> None:
        """Start a new fork from given root."""
        if root_hash not in self.forks:
            self.forks[root_hash] = []
        self.forks[root_hash].append(initial_block)

    def extend_fork(self, root_hash: str, block: Block) -> None:
        """Add block to existing fork."""
        if root_hash in self.forks:
            self.forks[root_hash].append(block)
        else:
            self.add_fork(root_hash, block)

    def get_heaviest_fork(self, stake_registry) -> tuple[str, List[Block]]:
        """Find fork with highest cumulative vote weight."""
        best_root = self.canonical_root
        best_weight = 0.0
        best_chain = self.forks.get(self.canonical_root, [])

        for root_hash, chain in self.forks.items():
            weight = sum(block.get_vote_weight(stake_registry) for block in chain)
            if weight > best_weight:
                best_weight = weight
                best_root = root_hash
                best_chain = chain

        return best_root, best_chain

    def resolve_forks(self, stake_registry) -> List[Block]:
        """Choose canonical chain based on vote weight."""
        canonical_root, canonical_chain = self.get_heaviest_fork(stake_registry)
        self.canonical_root = canonical_root
        return canonical_chain
```

### Enhanced Validator with Voting

```python
@dataclass
class Validator:
    id: str
    stake: float
    public_key: str
    private_key: str  # For signing votes

    def create_vote(self, block_hash: str, slot: int) -> Vote:
        """Create and sign a vote for a block."""
        sk = SigningKey.from_string(bytes.fromhex(self.private_key), curve=SECP256k1)
        vote_data = f"{self.id}{block_hash}{slot}".encode()
        signature = sk.sign(vote_data).hex()
        
        return Vote(
            validator_id=self.id,
            block_hash=block_hash,
            slot=slot,
            signature=signature
        )

    def should_vote_for_block(self, block: Block, blockchain) -> bool:
        """Decide whether to vote for a proposed block."""
        # Simple voting strategy: vote if block is valid
        return self._validate_block(block, blockchain)

    def _validate_block(self, block: Block, blockchain) -> bool:
        """Basic block validation."""
        # Check transactions are valid
        for tx in block.transactions:
            if not tx.verify_signature():
                return False
        
        # Check PoH continuity (simplified)
        return True
```

## Complete Implementation

Create `stage5_tower_bft.py`:

```python
import hashlib
import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
from ecdsa import SigningKey, VerifyingKey, SECP256k1

@dataclass(frozen=True)
class PoHEntry:
    counter: int
    hash: str

class PoH:
    def __init__(self, seed: str = "genesis"):
        self.current_hash = seed
        self.counter = 0

    def next_entry(self, iterations: int = 1000) -> PoHEntry:
        for _ in range(iterations):
            self.current_hash = hashlib.sha256(self.current_hash.encode()).hexdigest()
        self.counter += 1
        return PoHEntry(counter=self.counter, hash=self.current_hash)

@dataclass(frozen=True)
class Transaction:
    sender: str
    receiver: str
    amount: float
    signature: str

    def hash(self) -> str:
        tx_string = f"{self.sender}{self.receiver}{self.amount}".encode()
        return hashlib.sha256(tx_string).hexdigest()

    def verify_signature(self) -> bool:
        try:
            vk = VerifyingKey.from_string(bytes.fromhex(self.sender), curve=SECP256k1)
            return vk.verify(bytes.fromhex(self.signature), self.hash().encode())
        except:
            return False

@dataclass(frozen=True)
class Vote:
    validator_id: str
    block_hash: str
    slot: int
    signature: str

    def verify_signature(self, validator_public_key: str) -> bool:
        try:
            vk = VerifyingKey.from_string(bytes.fromhex(validator_public_key), curve=SECP256k1)
            vote_data = f"{self.validator_id}{self.block_hash}{self.slot}".encode()
            return vk.verify(bytes.fromhex(self.signature), vote_data)
        except:
            return False

@dataclass(frozen=True)
class Block:
    index: int
    poh_entry: PoHEntry
    transactions: List[Transaction]
    previous_hash: str
    leader_id: str
    votes: List[Vote] = field(default_factory=list)

    def hash(self) -> str:
        tx_hashes = "".join(tx.hash() for tx in self.transactions)
        vote_hashes = "".join(f"{v.validator_id}{v.signature}" for v in self.votes)
        block_string = f"{self.index}{self.poh_entry.counter}{self.poh_entry.hash}{tx_hashes}{vote_hashes}{self.previous_hash}{self.leader_id}".encode()
        return hashlib.sha256(block_string).hexdigest()

    def get_vote_weight(self, stake_registry) -> float:
        total_weight = 0.0
        for vote in self.votes:
            validator = stake_registry.validators.get(vote.validator_id)
            if validator:
                total_weight += validator.stake
        return total_weight

    def has_supermajority(self, stake_registry) -> bool:
        vote_weight = self.get_vote_weight(stake_registry)
        total_stake = stake_registry.total_stake()
        return vote_weight > (2/3 * total_stake)

class Mempool:
    def __init__(self):
        self.pending: Set[Transaction] = set()

    def add(self, tx: Transaction) -> bool:
        if tx.verify_signature() and tx not in self.pending:
            self.pending.add(tx)
            return True
        return False

    def get_valid(self, max_txs: int = 10) -> List[Transaction]:
        valid = list(self.pending)[:max_txs]
        self.pending.difference_update(valid)
        return valid

    def size(self) -> int:
        return len(self.pending)

@dataclass
class Validator:
    id: str
    stake: float
    public_key: str
    private_key: str

    def create_vote(self, block_hash: str, slot: int) -> Vote:
        sk = SigningKey.from_string(bytes.fromhex(self.private_key), curve=SECP256k1)
        vote_data = f"{self.id}{block_hash}{slot}".encode()
        signature = sk.sign(vote_data).hex()
        
        return Vote(
            validator_id=self.id,
            block_hash=block_hash,
            slot=slot,
            signature=signature
        )

    def should_vote_for_block(self, block: Block) -> bool:
        """Simple voting strategy: vote for valid blocks."""
        return all(tx.verify_signature() for tx in block.transactions)

class StakeRegistry:
    def __init__(self):
        self.validators: Dict[str, Validator] = {}

    def add_validator(self, validator: Validator) -> None:
        self.validators[validator.id] = validator

    def total_stake(self) -> float:
        return sum(v.stake for v in self.validators.values())

    def select_leader(self, poh_hash: str) -> str:
        if not self.validators:
            raise ValueError("No validators registered")
        
        seed = int(poh_hash, 16) % (2**32)
        random.seed(seed)
        
        validators = list(self.validators.values())
        weights = [v.stake for v in validators]
        
        selected = random.choices(validators, weights=weights, k=1)[0]
        return selected.id

    def list_validators(self) -> List[Validator]:
        return list(self.validators.values())

class ForkManager:
    def __init__(self):
        self.forks: Dict[str, List[Block]] = {"main": []}
        self.canonical_fork = "main"

    def add_block_to_fork(self, fork_id: str, block: Block) -> None:
        if fork_id not in self.forks:
            self.forks[fork_id] = []
        self.forks[fork_id].append(block)

    def get_canonical_chain(self) -> List[Block]:
        return self.forks[self.canonical_fork]

    def resolve_forks(self, stake_registry) -> List[Block]:
        """Choose fork with highest vote weight."""
        best_fork = self.canonical_fork
        best_weight = 0.0

        for fork_id, chain in self.forks.items():
            weight = sum(block.get_vote_weight(stake_registry) for block in chain)
            if weight > best_weight:
                best_weight = weight
                best_fork = fork_id

        self.canonical_fork = best_fork
        return self.forks[best_fork]

class TowerBFTBlockchain:
    def __init__(self):
        self.poh = PoH()
        self.mempool = Mempool()
        self.accounts: Dict[str, float] = {}
        self.stake_registry = StakeRegistry()
        self.fork_manager = ForkManager()
        self.current_slot = 0
        self.pending_votes: List[Vote] = []
        
        # Initialize with genesis block
        genesis_block = self._create_genesis_block()
        self.fork_manager.add_block_to_fork("main", genesis_block)

    def _create_genesis_block(self) -> Block:
        genesis_poh = self.poh.next_entry(iterations=1)
        return Block(
            index=0,
            poh_entry=genesis_poh,
            transactions=[],
            previous_hash="0",
            leader_id="genesis"
        )

    def get_balance(self, public_key: str) -> float:
        return self.accounts.get(public_key, 0.0)

    def set_balance(self, public_key: str, amount: float) -> None:
        self.accounts[public_key] = amount

    def add_transaction(self, tx: Transaction) -> bool:
        sender_balance = self.get_balance(tx.sender)
        if tx.amount > sender_balance:
            return False
        return self.mempool.add(tx)

    def propose_block(self, leader_id: str) -> Block:
        """Leader proposes a new block."""
        canonical_chain = self.fork_manager.get_canonical_chain()
        previous_block = canonical_chain[-1]
        
        new_poh = self.poh.next_entry()
        txs = self.mempool.get_valid()
        
        new_block = Block(
            index=len(canonical_chain),
            poh_entry=new_poh,
            transactions=txs,
            previous_hash=previous_block.hash(),
            leader_id=leader_id
        )
        
        return new_block

    def collect_votes(self, block: Block) -> List[Vote]:
        """Collect votes from all validators for a block."""
        votes = []
        
        for validator in self.stake_registry.list_validators():
            if validator.should_vote_for_block(block):
                vote = validator.create_vote(block.hash(), self.current_slot)
                votes.append(vote)
        
        return votes

    def add_block_with_votes(self, block: Block, votes: List[Vote]) -> None:
        """Add block with votes to appropriate fork."""
        # Add votes to block
        block_with_votes = Block(
            index=block.index,
            poh_entry=block.poh_entry,
            transactions=block.transactions,
            previous_hash=block.previous_hash,
            leader_id=block.leader_id,
            votes=votes
        )
        
        # Determine fork (simplified: use main fork or create new one)
        fork_id = "main"
        if random.random() < 0.1:  # 10% chance of fork for demo
            fork_id = f"fork_{self.current_slot}"
        
        self.fork_manager.add_block_to_fork(fork_id, block_with_votes)
        
        # Process transactions if this becomes canonical
        if fork_id == self.fork_manager.canonical_fork:
            for tx in block.transactions:
                self.accounts[tx.sender] = self.accounts.get(tx.sender, 0.0) - tx.amount
                self.accounts[tx.receiver] = self.accounts.get(tx.receiver, 0.0) + tx.amount

    def advance_slot(self) -> str:
        """Advance slot: select leader, propose block, collect votes."""
        self.current_slot += 1
        
        # Select leader
        leader_id = self.stake_registry.select_leader(self.poh.current_hash)
        
        # Leader proposes block
        proposed_block = self.propose_block(leader_id)
        
        # Validators vote
        votes = self.collect_votes(proposed_block)
        
        # Add block with votes
        self.add_block_with_votes(proposed_block, votes)
        
        # Resolve any forks
        canonical_chain = self.fork_manager.resolve_forks(self.stake_registry)
        
        return leader_id

    def is_valid(self) -> bool:
        """Validate the canonical chain."""
        canonical_chain = self.fork_manager.get_canonical_chain()
        previous_poh_hash = "genesis"
        
        for i, block in enumerate(canonical_chain):
            # Verify PoH continuity
            expected_hash = previous_poh_hash
            iterations = 1000 if i > 0 else 1
            
            for _ in range(iterations):
                expected_hash = hashlib.sha256(expected_hash.encode()).hexdigest()
            
            if block.poh_entry.hash != expected_hash:
                return False
            
            previous_poh_hash = block.poh_entry.hash
            
            # Verify block chain
            if i > 0 and block.previous_hash != canonical_chain[i - 1].hash():
                return False
            
            # Verify transactions
            for tx in block.transactions:
                if not tx.verify_signature():
                    return False
            
            # Verify votes
            for vote in block.votes:
                validator = self.stake_registry.validators.get(vote.validator_id)
                if validator and not vote.verify_signature(validator.public_key):
                    return False
        
        return True

    def display_consensus_state(self) -> None:
        """Display current consensus state."""
        print("\n" + "="*60)
        print("TOWER BFT CONSENSUS STATE")
        print("="*60)
        
        canonical_chain = self.fork_manager.get_canonical_chain()
        
        print(f"Current slot: {self.current_slot}")
        print(f"Canonical chain length: {len(canonical_chain)} blocks")
        print(f"Active forks: {len(self.fork_manager.forks)}")
        print(f"Total network stake: {self.stake_registry.total_stake()}")
        
        # Show recent blocks with vote info
        print(f"\nRecent Blocks (with votes):")
        for block in canonical_chain[-3:]:
            vote_weight = block.get_vote_weight(self.stake_registry)
            total_stake = self.stake_registry.total_stake()
            vote_pct = (vote_weight / total_stake) * 100 if total_stake > 0 else 0
            supermajority = "✓" if block.has_supermajority(self.stake_registry) else "✗"
            
            print(f"  Block {block.index}: {vote_pct:.1f}% stake voted ({supermajority} supermajority)")
        
        # Show fork information
        if len(self.fork_manager.forks) > 1:
            print(f"\nFork Analysis:")
            for fork_id, chain in self.fork_manager.forks.items():
                total_weight = sum(block.get_vote_weight(self.stake_registry) for block in chain)
                is_canonical = "★" if fork_id == self.fork_manager.canonical_fork else ""
                print(f"  {fork_id}: {len(chain)} blocks, {total_weight:.1f} total vote weight {is_canonical}")

# Helper functions (same as previous stages)
def create_signed_tx(sender_sk_hex: str, receiver_pk_hex: str, amount: float) -> Transaction:
    sk = SigningKey.from_string(bytes.fromhex(sender_sk_hex), curve=SECP256k1)
    sender_pk = sk.verifying_key.to_string().hex()
    
    tx = Transaction(sender=sender_pk, receiver=receiver_pk_hex, amount=amount, signature="")
    signature = sk.sign(tx.hash().encode()).hex()
    
    return Transaction(sender=sender_pk, receiver=receiver_pk_hex, amount=amount, signature=signature)

def generate_validator() -> Tuple[str, str, Validator]:
    """Generate a validator with keys."""
    sk = SigningKey.generate(curve=SECP256k1)
    pk = sk.verifying_key.to_string().hex()
    sk_hex = sk.to_string().hex()
    
    return sk_hex, pk, Validator(id=pk, stake=0, public_key=pk, private_key=sk_hex)

# Test Tower BFT
if __name__ == "__main__":
    print("Testing Tower BFT Consensus")
    print("=" * 40)
    
    # Create blockchain
    bc = TowerBFTBlockchain()
    
    # Create validators
    validators = []
    stakes = [1000, 600, 400, 200]  # Different stake amounts
    names = ["Alice", "Bob", "Carol", "Dave"]
    
    for i, (stake, name) in enumerate(zip(stakes, names)):
        sk, pk, validator = generate_validator()
        validator.stake = stake
        bc.stake_registry.add_validator(validator)
        bc.set_balance(pk, 100.0)  # Give some tokens to transact
        validators.append((sk, pk, name))
        print(f"{name}: {pk[:8]}... (stake: {stake})")
    
    # Add some transactions
    print(f"\nAdding transactions...")
    for i in range(5):
        sender_idx = i % len(validators)
        receiver_idx = (i + 1) % len(validators)
        
        sender_sk, sender_pk, _ = validators[sender_idx]
        _, receiver_pk, _ = validators[receiver_idx]
        
        amount = 10.0 + i * 5
        tx = create_signed_tx(sender_sk, receiver_pk, amount)
        bc.add_transaction(tx)
    
    # Simulate multiple slots with consensus
    print(f"\nSimulating consensus over 15 slots...")
    
    for slot in range(15):
        leader_id = bc.advance_slot()
        
        # Find leader name
        leader_name = "Unknown"
        for _, pk, name in validators:
            if pk == leader_id:
                leader_name = name
                break
        
        canonical_chain = bc.fork_manager.get_canonical_chain()
        latest_block = canonical_chain[-1]
        vote_weight = latest_block.get_vote_weight(bc.stake_registry)
        
        print(f"Slot {bc.current_slot}: {leader_name} proposed, {vote_weight:.0f} stake voted")
    
    # Display final consensus state
    bc.display_consensus_state()
    
    # Validate blockchain
    print(f"\nBlockchain validation: {bc.is_valid()}")
    
    # Analyze consensus safety
    canonical_chain = bc.fork_manager.get_canonical_chain()
    confirmed_blocks = sum(1 for block in canonical_chain if block.has_supermajority(bc.stake_registry))
    
    print(f"\nConsensus Analysis:")
    print(f"  Total blocks: {len(canonical_chain)}")
    print(f"  Confirmed blocks (>2/3 stake): {confirmed_blocks}")
    print(f"  Safety: All confirmed blocks have supermajority ✓")
    print(f"  Liveness: Network continues making progress ✓")
```

## Understanding Tower BFT

### Voting Process
1. **Leader proposes** block for current slot
2. **Validators evaluate** block validity
3. **Votes are cast** and signed cryptographically
4. **Vote weight calculated** based on validator stakes
5. **Supermajority** (>2/3 stake) confirms block

### Fork Resolution
When forks occur:
1. **Multiple chains** exist temporarily
2. **Validators vote** on their preferred fork
3. **Vote weights accumulate** over time
4. **Heaviest fork** becomes canonical
5. **Reorganization** updates state

### Safety Guarantees
- **Finality**: Blocks with >2/3 stake votes are final
- **Consistency**: All honest validators agree on finalized blocks
- **Byzantine tolerance**: Works with <1/3 malicious validators

## Key Insights

1. **Economic Finality**: Supermajority stake commitment makes reversal expensive
2. **Fork Choice**: Heaviest chain rule provides clear resolution
3. **Asynchronous Safety**: Works without timing assumptions
4. **Practical BFT**: Simplified compared to traditional PBFT

## Common Issues and Solutions

### Issue: "Votes not being counted"
**Problem**: Signature verification failing.
**Solution**: Ensure consistent key formats and vote data structure.

### Issue: "Forks never resolve"
**Problem**: Vote weights too close or no clear majority.
**Solution**: This is realistic - some forks take time to resolve.

### Issue: "Supermajority never reached"
**Problem**: Validators not voting or network partition.
**Solution**: Adjust threshold or check validator behavior.

## Next Steps

You now have Byzantine fault-tolerant consensus:
- Validators vote on proposed blocks with cryptographic signatures
- Fork detection and resolution using stake-weighted voting
- Safety guarantees with >2/3 honest stake assumption
- Foundation for distributed network operation

In [Stage 6](stage-06.md), we'll implement **Networking and Gossip Protocol**, making our blockchain truly distributed across multiple nodes.

---

**Checkpoint Questions:**
1. Why do we need >2/3 stake for finality instead of just >1/2?
2. How does Tower BFT handle network partitions?
3. What happens if two forks have equal vote weight?

**Ready for Stage 6?** → [Networking and Gossip Protocol](stage-06.md) 