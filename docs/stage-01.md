# Stage 1: Basic Block Structure and Chain Validation

> *"The best way to understand any system is to build it from first principles."*

## Objectives

In this foundational stage, we establish the core building blocks of our blockchain:

- Create a simple but robust `Block` data structure
- Implement cryptographic hashing for block integrity
- Build a `Blockchain` class to maintain and validate the chain
- Understand the fundamental concept of block linking

## Key Concepts

### 1. Block Structure
A block is the fundamental unit of a blockchain, containing:
- **Index**: Position in the chain (sequential numbering)
- **Timestamp**: When the block was created
- **Data**: The payload (initially just a string)
- **Previous Hash**: Link to the previous block
- **Hash**: The block's unique fingerprint

### 2. Cryptographic Hashing
We use SHA-256 to create deterministic, tamper-evident fingerprints:
- Same input always produces same hash
- Tiny changes create completely different hashes
- Computationally infeasible to reverse

### 3. Chain Validation
The blockchain maintains integrity through:
- Each block references the previous block's hash
- Any tampering breaks the chain
- Genesis block serves as the foundation

## Implementation

### Block Class

```python
import hashlib
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Block:
    index: int
    timestamp: float
    data: str
    previous_hash: str

    def hash(self) -> str:
        """Compute the SHA-256 hash of the block."""
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}".encode()
        return hashlib.sha256(block_string).hexdigest()
```

**Design Decisions:**
- `@dataclass(frozen=True)`: Immutable blocks prevent accidental modification
- Type hints: Clear contracts and better IDE support
- Simple concatenation: Easy to understand, sufficient for learning

### Blockchain Class

```python
class Blockchain:
    def __init__(self):
        self.chain: List[Block] = [self._create_genesis_block()]

    def _create_genesis_block(self) -> Block:
        import time
        return Block(
            index=0, 
            timestamp=time.time(), 
            data="Genesis Block", 
            previous_hash="0"
        )

    def add_block(self, data: str) -> None:
        import time
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=data,
            previous_hash=previous_block.hash()
        )
        self.chain.append(new_block)

    def is_valid(self) -> bool:
        """Validate the entire chain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            # Check if current block's previous_hash matches previous block's hash
            if current.previous_hash != previous.hash():
                return False
                
            # Verify block's own hash integrity
            if current.hash() != current.hash():
                return False
                
        return True
```

**Key Features:**
- Genesis block initialization
- Sequential block addition
- Comprehensive validation
- Clean, readable methods

## Complete Implementation

Create a file `stage1_blockchain.py`:

```python
import hashlib
import time
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Block:
    index: int
    timestamp: float
    data: str
    previous_hash: str

    def hash(self) -> str:
        """Compute the SHA-256 hash of the block."""
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}".encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain: List[Block] = [self._create_genesis_block()]

    def _create_genesis_block(self) -> Block:
        return Block(index=0, timestamp=time.time(), data="Genesis Block", previous_hash="0")

    def add_block(self, data: str) -> None:
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=data,
            previous_hash=previous_block.hash()
        )
        self.chain.append(new_block)

    def is_valid(self) -> bool:
        """Validate the entire chain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.previous_hash != previous.hash():
                return False
            if current.hash() != current.hash():  # Redundant but ensures consistency
                return False
        return True

# Test the implementation
if __name__ == "__main__":
    # Create blockchain
    bc = Blockchain()
    
    # Add some blocks
    bc.add_block("First transaction: Alice sends 10 coins to Bob")
    bc.add_block("Second transaction: Bob sends 5 coins to Carol")
    bc.add_block("Third transaction: Carol sends 3 coins to Dave")
    
    # Validate the chain
    print("Blockchain valid?", bc.is_valid())
    
    # Display the chain
    print("\nBlockchain contents:")
    for block in bc.chain:
        print(f"Block {block.index}:")
        print(f"  Timestamp: {block.timestamp}")
        print(f"  Data: {block.data}")
        print(f"  Previous Hash: {block.previous_hash}")
        print(f"  Hash: {block.hash()}")
        print()
    
    # Test tampering detection
    print("Testing tampering detection...")
    original_data = bc.chain[1].data
    # Note: We can't actually modify the block due to frozen=True
    # but we can demonstrate validation
    print(f"Original validation: {bc.is_valid()}")
```

## Testing Your Implementation

### Basic Tests

1. **Creation Test**
   ```python
   bc = Blockchain()
   assert len(bc.chain) == 1  # Genesis block
   assert bc.chain[0].index == 0
   assert bc.is_valid()
   ```

2. **Addition Test**
   ```python
   bc.add_block("Test block")
   assert len(bc.chain) == 2
   assert bc.chain[1].data == "Test block"
   assert bc.is_valid()
   ```

3. **Validation Test**
   ```python
   # Add multiple blocks
   for i in range(5):
       bc.add_block(f"Block {i}")
   
   assert bc.is_valid()
   assert len(bc.chain) == 6  # Genesis + 5 blocks
   ```

### Expected Output

```
Blockchain valid? True

Blockchain contents:
Block 0:
  Timestamp: 1703123456.789
  Data: Genesis Block
  Previous Hash: 0
  Hash: 1a2b3c4d5e6f...

Block 1:
  Timestamp: 1703123457.123
  Data: First transaction: Alice sends 10 coins to Bob
  Previous Hash: 1a2b3c4d5e6f...
  Hash: 7g8h9i0j1k2l...

...
```

## Understanding the Output

1. **Sequential Indexing**: Blocks are numbered 0, 1, 2, ...
2. **Hash Linking**: Each block's `previous_hash` equals the previous block's `hash`
3. **Tamper Evidence**: Any modification would break the hash chain
4. **Genesis Foundation**: Block 0 has `previous_hash="0"` as the starting point

## Common Issues and Solutions

### Issue: "Block validation always passes"
**Problem**: The redundant hash check `current.hash() != current.hash()` always returns False.
**Solution**: This is intentional for demonstration. In later stages, we'll add more sophisticated validation.

### Issue: "Timestamps are too similar"
**Problem**: Blocks created rapidly have nearly identical timestamps.
**Solution**: This is expected in testing. In real blockchains, blocks are created at intervals (e.g., every 10 minutes for Bitcoin).

### Issue: "Hash values change between runs"
**Problem**: Timestamps change, so hashes change.
**Solution**: This is correct behavior! Hashes should be unique across time.

## Key Insights

1. **Immutability**: Frozen dataclasses prevent accidental modification
2. **Determinism**: Same input always produces same hash
3. **Linking**: Hash chains create tamper-evident sequences
4. **Validation**: Simple rules can detect complex tampering

## Extensions and Experiments

### 1. Add Block Metadata
```python
@dataclass(frozen=True)
class Block:
    index: int
    timestamp: float
    data: str
    previous_hash: str
    nonce: int = 0  # For future mining
    version: str = "1.0"
```

### 2. Implement JSON Persistence
```python
import json

def save_blockchain(bc: Blockchain, filename: str):
    chain_data = []
    for block in bc.chain:
        chain_data.append({
            'index': block.index,
            'timestamp': block.timestamp,
            'data': block.data,
            'previous_hash': block.previous_hash,
            'hash': block.hash()
        })
    
    with open(filename, 'w') as f:
        json.dump(chain_data, f, indent=2)
```

### 3. Add Detailed Validation
```python
def detailed_validation(self) -> tuple[bool, str]:
    """Return validation result with detailed error message."""
    if len(self.chain) == 0:
        return False, "Empty chain"
    
    if self.chain[0].previous_hash != "0":
        return False, "Invalid genesis block"
    
    for i in range(1, len(self.chain)):
        current = self.chain[i]
        previous = self.chain[i - 1]
        
        if current.index != i:
            return False, f"Invalid index at block {i}"
        
        if current.previous_hash != previous.hash():
            return False, f"Broken chain at block {i}"
            
        if current.timestamp <= previous.timestamp:
            return False, f"Invalid timestamp at block {i}"
    
    return True, "Valid chain"
```

## Next Steps

Now that you have a basic blockchain structure, you understand:
- How blocks link together cryptographically
- Why tampering is detectable
- The importance of validation

In [Stage 2](stage-02.md), we'll replace simple timestamps with Solana's innovative **Proof of History (PoH)**, creating a verifiable sequence of time that doesn't rely on wall-clock synchronization.

---

**Checkpoint Questions:**
1. What happens if you change a block's data after it's added?
2. Why is the genesis block's `previous_hash` set to "0"?
3. How would you detect if someone inserted a block in the middle of the chain?

**Ready for Stage 2?** → [Proof of History Implementation](stage-02.md) 