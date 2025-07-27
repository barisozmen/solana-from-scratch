# Stage 4: Validators, Staking, and Leader Selection (Proof of Stake Basics)

> *"In Proof of Stake, we trust mathematics over energy consumption."* - Vitalik Buterin

## Objectives

In this stage, we transition from a single-node system to a multi-validator network with Proof of Stake (PoS) consensus:

- Implement validator registration with stake amounts
- Create stake-weighted leader selection using PoH for randomness
- Add slot-based block production (time-based intervals)
- Simulate multiple validators in a single process
- Establish the foundation for distributed consensus

## Why Proof of Stake?

### Problems with Proof of Work
- **Energy waste**: Massive computational overhead
- **Centralization**: Mining pools concentrate power
- **Scalability**: Limited transaction throughput
- **Barriers**: High hardware requirements

### PoS Advantages
- **Energy efficient**: No wasteful computation
- **Democratic**: Voting power proportional to stake
- **Scalable**: Faster block production
- **Economic security**: Validators risk their stake

## Key Concepts

### 1. Validators
Network participants who:
- **Stake tokens** to become eligible for block production
- **Validate transactions** and maintain network integrity
- **Earn rewards** for honest behavior
- **Risk slashing** for malicious actions

### 2. Stake-Weighted Selection
Leader selection based on:
- **Stake amount**: Higher stake = higher probability
- **Randomness**: PoH hash provides unpredictable seed
- **Fairness**: All validators have proportional chances
- **Determinism**: Same inputs produce same leader

### 3. Slots and Epochs
Time-based organization:
- **Slot**: Fixed time period for block production
- **Leader**: One validator selected per slot
- **Epoch**: Collection of slots (for future use)

## Implementation

### Validator Class

```python
@dataclass
class Validator:
    id: str          # Unique identifier (public key)
    stake: float     # Amount staked by validator
    public_key: str  # For verification (same as id)
    
    def __post_init__(self):
        """Validate constructor parameters."""
        if self.stake <= 0:
            raise ValueError("Stake must be positive")
        if not self.public_key:
            raise ValueError("Public key required")
```

### Stake Registry

```python
class StakeRegistry:
    def __init__(self):
        self.validators: Dict[str, Validator] = {}

    def add_validator(self, validator: Validator) -> None:
        """Register a new validator."""
        self.validators[validator.id] = validator
        print(f"Validator {validator.id[:8]}... registered with stake {validator.stake}")

    def total_stake(self) -> float:
        """Calculate total network stake."""
        return sum(v.stake for v in self.validators.values())

    def select_leader(self, poh_hash: str) -> str:
        """Select leader using stake-weighted randomness."""
        if not self.validators:
            raise ValueError("No validators registered")
        
        # Use PoH hash as deterministic random seed
        seed = int(poh_hash, 16) % (2**32)
        random.seed(seed)
        
        # Create stake-weighted selection
        validators = list(self.validators.values())
        weights = [v.stake for v in validators]
        
        # Use random.choices for weighted selection
        selected = random.choices(validators, weights=weights, k=1)[0]
        return selected.id

    def get_validator(self, validator_id: str) -> Validator:
        """Get validator by ID."""
        return self.validators.get(validator_id)
```

### Slot-Based Blockchain

```python
class Blockchain:
    def __init__(self):
        self.poh = PoH()
        self.chain: List[Block] = [self._create_genesis_block()]
        self.mempool = Mempool()
        self.accounts: Dict[str, float] = {}
        self.stake_registry = StakeRegistry()
        self.current_slot = 0

    def advance_slot(self) -> str:
        """Advance to next slot and select leader."""
        self.current_slot += 1
        
        # Generate new PoH entry (this advances time)
        new_poh = self.poh.next_entry()
        
        # Select leader based on PoH hash
        leader_id = self.stake_registry.select_leader(new_poh.hash)
        
        # Create block (simplified - in real system, only leader would do this)
        self._create_block_for_slot(leader_id, new_poh)
        
        return leader_id

    def _create_block_for_slot(self, leader_id: str, poh_entry: PoHEntry) -> None:
        """Create block for the current slot."""
        previous_block = self.chain[-1]
        
        # Get transactions from mempool
        txs = self.mempool.get_valid()
        
        # Process transactions
        for tx in txs:
            self.accounts[tx.sender] = self.accounts.get(tx.sender, 0.0) - tx.amount
            self.accounts[tx.receiver] = self.accounts.get(tx.receiver, 0.0) + tx.amount
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            poh_entry=poh_entry,
            transactions=txs,
            previous_hash=previous_block.hash()
        )
        
        self.chain.append(new_block)
        print(f"Slot {self.current_slot}: Leader {leader_id[:8]}... produced block {new_block.index}")
```

## Complete Implementation

Create `stage4_pos_blockchain.py`:

```python
import hashlib
import random
from dataclasses import dataclass
from typing import List, Dict, Set
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
class Block:
    index: int
    poh_entry: PoHEntry
    transactions: List[Transaction]
    previous_hash: str
    leader_id: str = ""  # Track which validator created this block

    def hash(self) -> str:
        tx_hashes = "".join(tx.hash() for tx in self.transactions)
        block_string = f"{self.index}{self.poh_entry.counter}{self.poh_entry.hash}{tx_hashes}{self.previous_hash}{self.leader_id}".encode()
        return hashlib.sha256(block_string).hexdigest()

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

    def __post_init__(self):
        if self.stake <= 0:
            raise ValueError("Stake must be positive")

class StakeRegistry:
    def __init__(self):
        self.validators: Dict[str, Validator] = {}

    def add_validator(self, validator: Validator) -> None:
        self.validators[validator.id] = validator

    def total_stake(self) -> float:
        return sum(v.stake for v in self.validators.values())

    def select_leader(self, poh_hash: str) -> str:
        """Select leader using stake-weighted randomness."""
        if not self.validators:
            raise ValueError("No validators registered")
        
        # Use PoH hash for deterministic randomness
        seed = int(poh_hash, 16) % (2**32)
        random.seed(seed)
        
        # Stake-weighted selection
        validators = list(self.validators.values())
        weights = [v.stake for v in validators]
        
        selected = random.choices(validators, weights=weights, k=1)[0]
        return selected.id

    def get_validator_stake(self, validator_id: str) -> float:
        validator = self.validators.get(validator_id)
        return validator.stake if validator else 0.0

    def list_validators(self) -> List[Validator]:
        return list(self.validators.values())

class Blockchain:
    def __init__(self):
        self.poh = PoH()
        self.chain: List[Block] = [self._create_genesis_block()]
        self.mempool = Mempool()
        self.accounts: Dict[str, float] = {}
        self.stake_registry = StakeRegistry()
        self.current_slot = 0
        self.leader_history: List[str] = []

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

    def advance_slot(self) -> str:
        """Advance to next slot, select leader, and create block."""
        self.current_slot += 1
        
        # Generate new PoH entry
        new_poh = self.poh.next_entry()
        
        # Select leader
        leader_id = self.stake_registry.select_leader(new_poh.hash)
        self.leader_history.append(leader_id)
        
        # Create block
        previous_block = self.chain[-1]
        txs = self.mempool.get_valid()
        
        # Process transactions
        for tx in txs:
            self.accounts[tx.sender] = self.accounts.get(tx.sender, 0.0) - tx.amount
            self.accounts[tx.receiver] = self.accounts.get(tx.receiver, 0.0) + tx.amount
        
        new_block = Block(
            index=len(self.chain),
            poh_entry=new_poh,
            transactions=txs,
            previous_hash=previous_block.hash(),
            leader_id=leader_id
        )
        
        self.chain.append(new_block)
        return leader_id

    def is_valid(self) -> bool:
        """Validate blockchain including PoH and transactions."""
        previous_poh_hash = "genesis"
        
        for i, block in enumerate(self.chain):
            # Verify PoH continuity
            expected_hash = previous_poh_hash
            iterations = 1000 if i > 0 else 1
            
            for _ in range(iterations):
                expected_hash = hashlib.sha256(expected_hash.encode()).hexdigest()
            
            if block.poh_entry.hash != expected_hash:
                return False
            
            previous_poh_hash = block.poh_entry.hash
            
            # Verify block chain
            if i > 0 and block.previous_hash != self.chain[i - 1].hash():
                return False
            
            # Verify transactions
            for tx in block.transactions:
                if not tx.verify_signature():
                    return False
        
        return True

    def analyze_leader_distribution(self) -> Dict[str, int]:
        """Analyze how often each validator was selected as leader."""
        distribution = {}
        for leader_id in self.leader_history:
            distribution[leader_id] = distribution.get(leader_id, 0) + 1
        return distribution

    def display_state(self) -> None:
        """Display current blockchain state."""
        print("\n" + "="*60)
        print("PROOF OF STAKE BLOCKCHAIN STATE")
        print("="*60)
        
        print(f"Current slot: {self.current_slot}")
        print(f"Chain length: {len(self.chain)} blocks")
        print(f"Mempool size: {self.mempool.size()} pending transactions")
        print(f"Total network stake: {self.stake_registry.total_stake()}")
        
        print("\nValidators:")
        for validator in self.stake_registry.list_validators():
            stake_pct = (validator.stake / self.stake_registry.total_stake()) * 100
            print(f"  {validator.id[:8]}...: {validator.stake} tokens ({stake_pct:.1f}%)")
        
        print("\nLeader Distribution:")
        distribution = self.analyze_leader_distribution()
        for leader_id, count in distribution.items():
            if leader_id != "genesis":
                pct = (count / len(self.leader_history)) * 100 if self.leader_history else 0
                print(f"  {leader_id[:8]}...: {count} blocks ({pct:.1f}%)")
        
        print("\nAccount Balances:")
        for account, balance in self.accounts.items():
            print(f"  {account[:8]}...: {balance}")

# Helper functions
def create_signed_tx(sender_sk_hex: str, receiver_pk_hex: str, amount: float) -> Transaction:
    sk = SigningKey.from_string(bytes.fromhex(sender_sk_hex), curve=SECP256k1)
    sender_pk = sk.verifying_key.to_string().hex()
    
    tx = Transaction(sender=sender_pk, receiver=receiver_pk_hex, amount=amount, signature="")
    signature = sk.sign(tx.hash().encode()).hex()
    
    return Transaction(sender=sender_pk, receiver=receiver_pk_hex, amount=amount, signature=signature)

def generate_keypair() -> tuple[str, str]:
    sk = SigningKey.generate(curve=SECP256k1)
    pk = sk.verifying_key.to_string().hex()
    sk_hex = sk.to_string().hex()
    return sk_hex, pk

# Test the PoS system
if __name__ == "__main__":
    print("Testing Proof of Stake Blockchain")
    print("=" * 40)
    
    # Create blockchain
    bc = Blockchain()
    
    # Create validators with different stake amounts
    validators = []
    
    # High-stake validator
    val1_sk, val1_pk = generate_keypair()
    val1 = Validator(id=val1_pk, stake=1000.0, public_key=val1_pk)
    bc.stake_registry.add_validator(val1)
    validators.append((val1_sk, val1_pk, "Alice (High Stake)"))
    
    # Medium-stake validator
    val2_sk, val2_pk = generate_keypair()
    val2 = Validator(id=val2_pk, stake=500.0, public_key=val2_pk)
    bc.stake_registry.add_validator(val2)
    validators.append((val2_sk, val2_pk, "Bob (Medium Stake)"))
    
    # Low-stake validator
    val3_sk, val3_pk = generate_keypair()
    val3 = Validator(id=val3_pk, stake=100.0, public_key=val3_pk)
    bc.stake_registry.add_validator(val3)
    validators.append((val3_sk, val3_pk, "Carol (Low Stake)"))
    
    # Set initial balances
    bc.set_balance(val1_pk, 100.0)
    bc.set_balance(val2_pk, 100.0)
    bc.set_balance(val3_pk, 100.0)
    
    print(f"\nValidators registered:")
    for _, pk, name in validators:
        stake = bc.stake_registry.get_validator_stake(pk)
        print(f"  {name}: {pk[:8]}... (stake: {stake})")
    
    # Add some transactions
    print(f"\nAdding transactions...")
    tx1 = create_signed_tx(val1_sk, val2_pk, 25.0)
    tx2 = create_signed_tx(val2_sk, val3_pk, 15.0)
    tx3 = create_signed_tx(val3_sk, val1_pk, 10.0)
    
    bc.add_transaction(tx1)
    bc.add_transaction(tx2)
    bc.add_transaction(tx3)
    
    # Simulate multiple slots
    print(f"\nSimulating 20 slots...")
    
    for slot in range(20):
        leader_id = bc.advance_slot()
        
        # Find validator name for display
        validator_name = "Unknown"
        for _, pk, name in validators:
            if pk == leader_id:
                validator_name = name
                break
        
        print(f"Slot {bc.current_slot}: {validator_name} ({leader_id[:8]}...)")
        
        # Occasionally add more transactions
        if slot % 5 == 0 and slot > 0:
            # Random transaction between validators
            sender_idx = random.randint(0, 2)
            receiver_idx = (sender_idx + 1) % 3
            
            sender_sk, sender_pk, _ = validators[sender_idx]
            _, receiver_pk, _ = validators[receiver_idx]
            
            amount = random.uniform(1.0, 10.0)
            
            if bc.get_balance(sender_pk) >= amount:
                tx = create_signed_tx(sender_sk, receiver_pk, amount)
                bc.add_transaction(tx)
                print(f"  → Added transaction: {amount:.1f} tokens")
    
    # Display final state
    bc.display_state()
    
    # Validate blockchain
    print(f"\nBlockchain validation: {bc.is_valid()}")
    
    # Analyze stake vs selection frequency
    print(f"\nStake vs Selection Analysis:")
    distribution = bc.analyze_leader_distribution()
    total_selections = sum(distribution.values()) - distribution.get("genesis", 0)
    
    for validator in bc.stake_registry.list_validators():
        selections = distribution.get(validator.id, 0)
        expected_pct = (validator.stake / bc.stake_registry.total_stake()) * 100
        actual_pct = (selections / total_selections) * 100 if total_selections > 0 else 0
        
        print(f"  {validator.id[:8]}...: Expected {expected_pct:.1f}%, Actual {actual_pct:.1f}%")
```

## Understanding Leader Selection

### Stake-Weighted Probability

The probability of being selected as leader is proportional to stake:

```
P(validator) = validator_stake / total_stake
```

### Deterministic Randomness

Using PoH hash ensures:
- **Unpredictable**: Cannot predict future leaders
- **Verifiable**: Anyone can verify leader selection
- **Consistent**: All nodes agree on the same leader

### Example Selection Process

```python
# Given validators:
# Alice: 1000 stake (62.5% chance)
# Bob:   500 stake (31.25% chance)  
# Carol: 100 stake (6.25% chance)

poh_hash = "a1b2c3d4e5f6..."
seed = int(poh_hash, 16) % (2**32)
random.seed(seed)

validators = [alice, bob, carol]
weights = [1000, 500, 100]

# Weighted random selection
leader = random.choices(validators, weights=weights, k=1)[0]
```

## Testing Leader Distribution

### Statistical Analysis

```python
def test_leader_selection_fairness():
    """Test that leader selection is statistically fair."""
    
    bc = Blockchain()
    
    # Add validators with known stakes
    val_a = Validator("validator_a", 600, "key_a")
    val_b = Validator("validator_b", 300, "key_b") 
    val_c = Validator("validator_c", 100, "key_c")
    
    bc.stake_registry.add_validator(val_a)
    bc.stake_registry.add_validator(val_b)
    bc.stake_registry.add_validator(val_c)
    
    # Run many selections
    selections = {}
    for _ in range(1000):
        leader = bc.advance_slot()
        selections[leader] = selections.get(leader, 0) + 1
    
    # Calculate expected vs actual
    total_stake = bc.stake_registry.total_stake()
    
    for validator in [val_a, val_b, val_c]:
        expected = (validator.stake / total_stake) * 1000
        actual = selections.get(validator.id, 0)
        deviation = abs(expected - actual) / expected
        
        print(f"{validator.id}: Expected {expected:.0f}, Actual {actual}, Deviation {deviation:.2%}")
        
        # Should be within 10% for large samples
        assert deviation < 0.1, f"Selection bias detected for {validator.id}"

if __name__ == "__main__":
    test_leader_selection_fairness()
```

## Key Insights

1. **Economic Security**: Validators risk their stake, aligning incentives
2. **Proportional Representation**: Stake determines influence fairly
3. **Randomness**: PoH provides unpredictable but verifiable leader selection
4. **Efficiency**: No wasteful computation like Proof of Work

## Common Issues and Solutions

### Issue: "Same validator selected repeatedly"
**Problem**: Random selection can have streaks.
**Solution**: This is normal - with enough slots, distribution equalizes.

### Issue: "Low-stake validators never selected"
**Problem**: Very small stakes have tiny probabilities.
**Solution**: Consider minimum stake requirements or delegation.

### Issue: "Leader selection is predictable"
**Problem**: Using weak randomness source.
**Solution**: PoH provides cryptographically strong randomness.

## Extensions and Experiments

### 1. Validator Rewards
```python
def distribute_rewards(self, block: Block, reward: float):
    """Distribute block rewards to the leader."""
    leader_id = block.leader_id
    if leader_id in self.accounts:
        self.accounts[leader_id] += reward
        print(f"Reward {reward} distributed to {leader_id[:8]}...")
```

### 2. Delegation System
```python
@dataclass
class DelegatedStake:
    delegator: str    # Who delegated
    validator: str    # Who received delegation
    amount: float     # How much was delegated

class StakeRegistry:
    def __init__(self):
        self.validators: Dict[str, Validator] = {}
        self.delegations: List[DelegatedStake] = []
    
    def delegate_stake(self, delegator: str, validator: str, amount: float):
        """Delegate stake to a validator."""
        delegation = DelegatedStake(delegator, validator, amount)
        self.delegations.append(delegation)
    
    def get_total_stake(self, validator_id: str) -> float:
        """Get validator's own stake plus delegated stake."""
        own_stake = self.validators[validator_id].stake
        delegated = sum(d.amount for d in self.delegations if d.validator == validator_id)
        return own_stake + delegated
```

### 3. Slashing Conditions
```python
class SlashingConditions:
    def __init__(self, stake_registry: StakeRegistry):
        self.stake_registry = stake_registry
        self.slashed_validators: Set[str] = set()
    
    def slash_validator(self, validator_id: str, reason: str, amount: float):
        """Slash a validator's stake for misbehavior."""
        validator = self.stake_registry.validators.get(validator_id)
        if validator and validator_id not in self.slashed_validators:
            validator.stake = max(0, validator.stake - amount)
            self.slashed_validators.add(validator_id)
            print(f"Validator {validator_id[:8]}... slashed {amount} for {reason}")
```

## Next Steps

You now have a working Proof of Stake system:
- Validators with economic stake in the network
- Fair, stake-weighted leader selection using PoH randomness
- Slot-based block production with clear timing
- Foundation for Byzantine fault tolerance

In [Stage 5](stage-05.md), we'll implement **Tower BFT Consensus**, adding voting mechanisms and fork resolution to handle network disagreements.

---

**Checkpoint Questions:**
1. Why is stake-weighted selection more fair than random selection?
2. How does PoH provide randomness for leader selection?
3. What happens if a validator with low stake never gets selected?

**Ready for Stage 5?** → [Consensus with Tower BFT](stage-05.md) 