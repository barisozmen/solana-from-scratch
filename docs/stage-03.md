# Stage 3: Transactions and Instructions - Solana's Transaction Model

> *"A transaction is a collection of instructions that are executed atomically."* - [Solana Documentation](https://solana.com/docs/core/transactions)

## Objectives

In this crucial stage, we implement Solana's unique transaction and instruction model, which differs significantly from other blockchains:

- Understand Solana's atomic transaction execution model
- Implement instructions as the basic unit of program execution
- Create proper transaction structure with multiple instructions
- Add account metadata for precise access control
- Integrate transaction fees and compute budget management

## Why Solana's Transaction Model is Different

### Traditional Transaction Limitations
Most blockchains have simple transactions:
- **One operation per transaction**: Transfer from A to B
- **Limited composability**: Can't combine multiple operations
- **State conflicts**: Difficulty determining what accounts are accessed
- **Gas estimation**: Hard to predict execution costs

### Solana's Instruction-Based Model
Based on [official Solana transaction documentation](https://solana.com/docs/core/transactions):
- **Multiple instructions per transaction**: Atomic execution of complex operations
- **Explicit account lists**: Every account accessed is declared upfront
- **Program composition**: Instructions can invoke multiple programs
- **Deterministic fees**: Clear compute budget and fee structure
- **Parallel execution**: Account access patterns enable concurrent processing

## Key Concepts from Solana Documentation

### 1. Transaction Structure
```rust
pub struct Transaction {
    pub signatures: Vec<Signature>,        // Ed25519 signatures
    pub message: Message,                  // Transaction message
}

pub struct Message {
    pub header: MessageHeader,             // Account access metadata
    pub account_keys: Vec<Pubkey>,        // All accounts referenced
    pub recent_blockhash: Hash,           // Recent blockhash for replay protection
    pub instructions: Vec<CompiledInstruction>, // Instructions to execute
}
```

### 2. Instruction Model
Each instruction specifies:
- **Program ID**: Which program to execute
- **Accounts**: Array of all accounts the instruction will access
- **Instruction Data**: Program-specific parameters

### 3. Account Access Patterns
- **Read-only accounts**: Can be accessed by multiple instructions simultaneously
- **Writable accounts**: Require exclusive access during execution
- **Signer accounts**: Must provide valid Ed25519 signatures

### 4. Transaction Fees
Based on [Solana's fee structure](https://solana.com/docs/core/fees):
- **Base fee**: Per signature (5,000 lamports per signature)
- **Compute budget**: Computational units consumed
- **Priority fees**: Optional higher fees for faster processing

## Implementation

### Updated Transaction Structure

```python
from dataclasses import dataclass, field
from typing import List, Dict, Set
import hashlib
from ecdsa import SigningKey, VerifyingKey, SECP256k1

@dataclass
class MessageHeader:
    num_required_signatures: int
    num_readonly_signed_accounts: int
    num_readonly_unsigned_accounts: int

@dataclass
class CompiledInstruction:
    program_id_index: int
    accounts: List[int]
    data: bytes

@dataclass
class TransactionMessage:
    header: MessageHeader
    account_keys: List[str]
    recent_blockhash: str
    instructions: List[CompiledInstruction]
    
    def serialize(self) -> bytes:
        data = []
        data.append(self.header.num_required_signatures.to_bytes(1, 'little'))
        data.append(self.header.num_readonly_signed_accounts.to_bytes(1, 'little'))
        data.append(self.header.num_readonly_unsigned_accounts.to_bytes(1, 'little'))
        data.append(len(self.account_keys).to_bytes(1, 'little'))
        
        for key in self.account_keys:
            data.append(bytes.fromhex(key))
        
        data.append(bytes.fromhex(self.recent_blockhash))
        data.append(len(self.instructions).to_bytes(1, 'little'))
        
        for instruction in self.instructions:
            data.append(instruction.program_id_index.to_bytes(1, 'little'))
            data.append(len(instruction.accounts).to_bytes(1, 'little'))
            data.extend(acc.to_bytes(1, 'little') for acc in instruction.accounts)
            data.append(len(instruction.data).to_bytes(2, 'little'))
            data.append(instruction.data)
        
        return b''.join(data)

@dataclass
class SolanaTransaction:
    signatures: List[str]
    message: TransactionMessage
    
    def hash(self) -> str:
        return hashlib.sha256(self.message.serialize()).hexdigest()
    
    def verify_signatures(self) -> bool:
        message_data = self.message.serialize()
        
        if len(self.signatures) < self.message.header.num_required_signatures:
            return False
        
        for i in range(self.message.header.num_required_signatures):
            if i >= len(self.signatures) or i >= len(self.message.account_keys):
                return False
            
            try:
                signer_pubkey = self.message.account_keys[i]
                signature = self.signatures[i]
                
                vk = VerifyingKey.from_string(bytes.fromhex(signer_pubkey), curve=SECP256k1)
                if not vk.verify(bytes.fromhex(signature), message_data):
                    return False
            except:
                return False
        
        return True
    
    def get_fee_payer(self) -> str:
        if not self.message.account_keys:
            raise ValueError("No accounts in transaction")
        return self.message.account_keys[0]
    
    def calculate_fee(self) -> int:
        base_fee = len(self.signatures) * 5000
        compute_units = sum(len(instr.data) * 100 for instr in self.message.instructions)
        compute_fee = min(compute_units, 200000)
        return base_fee + compute_fee

@dataclass
class AccountMeta:
    pubkey: str
    is_signer: bool
    is_writable: bool

@dataclass
class Instruction:
    program_id: str
    accounts: List[AccountMeta]
    data: bytes

class TransactionBuilder:
    def __init__(self, fee_payer: str, recent_blockhash: str):
        self.fee_payer = fee_payer
        self.recent_blockhash = recent_blockhash
        self.instructions: List[Instruction] = []
    
    def add_instruction(self, instruction: Instruction) -> 'TransactionBuilder':
        self.instructions.append(instruction)
        return self
    
    def build(self) -> TransactionMessage:
        # Collect all accounts
        all_accounts = {self.fee_payer}
        
        for instruction in self.instructions:
            all_accounts.add(instruction.program_id)
            for account in instruction.accounts:
                all_accounts.add(account.pubkey)
        
        # Determine account properties
        signer_accounts = {self.fee_payer}
        writable_accounts = {self.fee_payer}
        
        for instruction in self.instructions:
            for account in instruction.accounts:
                if account.is_signer:
                    signer_accounts.add(account.pubkey)
                if account.is_writable:
                    writable_accounts.add(account.pubkey)
        
        # Order accounts
        account_keys = []
        
        # Required signers (writable signers)
        required_signers = sorted(signer_accounts & writable_accounts)
        account_keys.extend(required_signers)
        
        # Readonly signers
        readonly_signers = sorted(signer_accounts - writable_accounts)
        account_keys.extend(readonly_signers)
        
        # Writable non-signers
        writable_non_signers = sorted((writable_accounts - signer_accounts))
        account_keys.extend(writable_non_signers)
        
        # Readonly non-signers
        readonly_non_signers = sorted(all_accounts - set(account_keys))
        account_keys.extend(readonly_non_signers)
        
        # Create index mapping
        account_index = {key: i for i, key in enumerate(account_keys)}
        
        # Compile instructions
        compiled_instructions = []
        for instruction in self.instructions:
            compiled_instruction = CompiledInstruction(
                program_id_index=account_index[instruction.program_id],
                accounts=[account_index[acc.pubkey] for acc in instruction.accounts],
                data=instruction.data
            )
            compiled_instructions.append(compiled_instruction)
        
        # Create header
        header = MessageHeader(
            num_required_signatures=len(required_signers),
            num_readonly_signed_accounts=len(readonly_signers),
            num_readonly_unsigned_accounts=len(readonly_non_signers)
        )
        
        return TransactionMessage(
            header=header,
            account_keys=account_keys,
            recent_blockhash=self.recent_blockhash,
            instructions=compiled_instructions
        )

class SystemProgram:
    PROGRAM_ID = "11111111111111111111111111111111"
    
    @staticmethod
    def transfer(from_pubkey: str, to_pubkey: str, lamports: int) -> Instruction:
        data = bytearray()
        data.extend([2])  # Transfer instruction
        data.extend(lamports.to_bytes(8, 'little'))
        
        return Instruction(
            program_id=SystemProgram.PROGRAM_ID,
            accounts=[
                AccountMeta(from_pubkey, is_signer=True, is_writable=True),
                AccountMeta(to_pubkey, is_signer=False, is_writable=True)
            ],
            data=bytes(data)
        )

class SolanaMempool:
    def __init__(self):
        self.pending: Dict[str, SolanaTransaction] = {}
        self.fee_priority: List[Tuple[int, str]] = []
        self.account_locks: Dict[str, Set[str]] = {}
    
    def add_transaction(self, tx: SolanaTransaction) -> bool:
        tx_hash = tx.hash()
        
        if tx_hash in self.pending:
            return False
        
        if not tx.verify_signatures():
            return False
        
        if self._has_account_conflicts(tx):
            return False
        
        self.pending[tx_hash] = tx
        fee = tx.calculate_fee()
        self.fee_priority.append((fee, tx_hash))
        self.fee_priority.sort(reverse=True)
        
        self._track_account_usage(tx, tx_hash)
        return True
    
    def _has_account_conflicts(self, tx: SolanaTransaction) -> bool:
        writable_accounts = self._get_writable_accounts(tx)
        
        for account in writable_accounts:
            if account in self.account_locks and self.account_locks[account]:
                return True
        
        return False
    
    def _get_writable_accounts(self, tx: SolanaTransaction) -> Set[str]:
        writable = {tx.get_fee_payer()}
        
        header = tx.message.header
        num_signers = header.num_required_signatures + header.num_readonly_signed_accounts
        
        for i, account_key in enumerate(tx.message.account_keys):
            if i < num_signers - header.num_readonly_signed_accounts or (
                i >= num_signers and 
                i < len(tx.message.account_keys) - header.num_readonly_unsigned_accounts
            ):
                writable.add(account_key)
        
        return writable
    
    def _track_account_usage(self, tx: SolanaTransaction, tx_hash: str) -> None:
        for account_key in tx.message.account_keys:
            if account_key not in self.account_locks:
                self.account_locks[account_key] = set()
            self.account_locks[account_key].add(tx_hash)
    
    def get_transactions_for_block(self, max_transactions: int = 10) -> List[SolanaTransaction]:
        selected = []
        selected_hashes = set()
        used_accounts = set()
        
        for fee, tx_hash in self.fee_priority:
            if len(selected) >= max_transactions:
                break
            
            if tx_hash in selected_hashes or tx_hash not in self.pending:
                continue
            
            tx = self.pending[tx_hash]
            tx_writable_accounts = self._get_writable_accounts(tx)
            
            if tx_writable_accounts & used_accounts:
                continue
            
            selected.append(tx)
            selected_hashes.add(tx_hash)
            used_accounts.update(tx_writable_accounts)
        
        for tx in selected:
            self._remove_transaction(tx.hash())
        
        return selected
    
    def _remove_transaction(self, tx_hash: str) -> None:
        if tx_hash in self.pending:
            tx = self.pending[tx_hash]
            del self.pending[tx_hash]
            
            self.fee_priority = [(fee, h) for fee, h in self.fee_priority if h != tx_hash]
            
            for account_key in tx.message.account_keys:
                if account_key in self.account_locks:
                    self.account_locks[account_key].discard(tx_hash)
                    if not self.account_locks[account_key]:
                        del self.account_locks[account_key]
    
    def size(self) -> int:
        return len(self.pending)

@dataclass(frozen=True)
class SolanaBlock:
    index: int
    poh_entry: PoHEntry
    transactions: List[SolanaTransaction]
    previous_hash: str

    def hash(self) -> str:
        tx_hashes = "".join(tx.hash() for tx in self.transactions)
        block_string = f"{self.index}{self.poh_entry.counter}{self.poh_entry.hash}{tx_hashes}{self.previous_hash}".encode()
        return hashlib.sha256(block_string).hexdigest()

class SolanaBlockchain(BaseBlockchain):
    def __init__(self):
        super().__init__()
        self.mempool = SolanaMempool()
        self.accounts: Dict[str, int] = {}  # pubkey -> lamport balance
        self.recent_blockhashes: List[str] = []
    
    def get_recent_blockhash(self) -> str:
        """Get recent blockhash for transaction creation."""
        if self.recent_blockhashes:
            return self.recent_blockhashes[-1]
        return "1111111111111111111111111111111111111111111111111111111111111111"
    
    def add_transaction(self, tx: SolanaTransaction) -> bool:
        """Add transaction to mempool."""
        return self.mempool.add_transaction(tx)
    
    def create_block(self) -> None:
        """Create new block with transactions from mempool."""
        previous_block = self.chain[-1] if hasattr(self, 'chain') else None
        new_poh = self.poh.next_entry()
        
        # Get transactions for block
        transactions = self.mempool.get_transactions_for_block()
        
        # Process transactions
        for tx in transactions:
            self._process_transaction(tx)
        
        # Create block
        new_block = SolanaBlock(
            index=len(getattr(self, 'chain', [])),
            poh_entry=new_poh,
            transactions=transactions,
            previous_hash=previous_block.hash() if previous_block else "0"
        )
        
        if not hasattr(self, 'chain'):
            self.chain = []
        self.chain.append(new_block)
        
        # Update recent blockhashes
        self.recent_blockhashes.append(new_block.hash())
        if len(self.recent_blockhashes) > 150:  # Keep last 150 blocks
            self.recent_blockhashes.pop(0)
        
        print(f"Created block {new_block.index} with {len(transactions)} transactions")
    
    def _process_transaction(self, tx: SolanaTransaction) -> None:
        """Process a transaction (simplified execution)."""
        # Charge fees
        fee_payer = tx.get_fee_payer()
        fee = tx.calculate_fee()
        
        if fee_payer in self.accounts:
            self.accounts[fee_payer] -= fee
        
        # Process instructions (simplified - just system transfers)
        for instruction in tx.message.instructions:
            if instruction.program_id_index < len(tx.message.account_keys):
                program_id = tx.message.account_keys[instruction.program_id_index]
                if program_id == SystemProgram.PROGRAM_ID and len(instruction.data) >= 9:
                    self._process_system_instruction(instruction, tx.message)
    
    def _process_system_instruction(self, instruction: CompiledInstruction, message: TransactionMessage) -> None:
        """Process system program instruction."""
        if len(instruction.data) < 1:
            return
        
        instruction_type = instruction.data[0]
        
        if instruction_type == 2 and len(instruction.accounts) >= 2:  # Transfer
            from_idx = instruction.accounts[0]
            to_idx = instruction.accounts[1]
            
            if from_idx < len(message.account_keys) and to_idx < len(message.account_keys):
                from_account = message.account_keys[from_idx]
                to_account = message.account_keys[to_idx]
                
                if len(instruction.data) >= 9:
                    amount = int.from_bytes(instruction.data[1:9], 'little')
                    
                    # Ensure accounts exist
                    if from_account not in self.accounts:
                        self.accounts[from_account] = 0
                    if to_account not in self.accounts:
                        self.accounts[to_account] = 0
                    
                    # Transfer
                    if self.accounts[from_account] >= amount:
                        self.accounts[from_account] -= amount
                        self.accounts[to_account] += amount
    
    def get_balance(self, pubkey: str) -> int:
        """Get account balance in lamports."""
        return self.accounts.get(pubkey, 0)
    
    def set_balance(self, pubkey: str, lamports: int) -> None:
        """Set account balance (for testing)."""
        self.accounts[pubkey] = lamports
    
    def display_state(self) -> None:
        """Display blockchain state."""
        print(f"\n{'='*50}")
        print("SOLANA BLOCKCHAIN STATE")
        print(f"{'='*50}")
        
        if hasattr(self, 'chain'):
            print(f"Chain length: {len(self.chain)} blocks")
            print(f"Mempool size: {self.mempool.size()} transactions")
            
            print(f"\nAccount Balances:")
            for pubkey, balance in self.accounts.items():
                sol_balance = balance / 1_000_000_000
                print(f"  {pubkey[:8]}...: {balance:,} lamports ({sol_balance:.6f} SOL)")
            
            print(f"\nRecent Blocks:")
            for block in getattr(self, 'chain', [])[-3:]:
                print(f"  Block {block.index}: {len(block.transactions)} transactions")

def sign_transaction(message: TransactionMessage, signers: List[SigningKey]) -> SolanaTransaction:
    """Sign a transaction message."""
    message_data = message.serialize()
    signatures = []
    
    for signer in signers:
        signature = signer.sign(message_data).hex()
        signatures.append(signature)
    
    return SolanaTransaction(signatures=signatures, message=message)

def generate_keypair() -> Tuple[SigningKey, str]:
    """Generate keypair and return private key and public key hex."""
    sk = SigningKey.generate(curve=SECP256k1)
    pk = sk.verifying_key.to_string().hex()
    return sk, pk

# Test the Solana transaction system
if __name__ == "__main__":
    print("Testing Solana Transaction System")
    print("=" * 40)
    
    # Create blockchain
    bc = SolanaBlockchain()
    
    # Generate keypairs
    alice_sk, alice_pk = generate_keypair()
    bob_sk, bob_pk = generate_keypair()
    charlie_sk, charlie_pk = generate_keypair()
    
    print(f"Alice: {alice_pk[:8]}...")
    print(f"Bob: {bob_pk[:8]}...")
    print(f"Charlie: {charlie_pk[:8]}...")
    
    # Set initial balances
    bc.set_balance(alice_pk, 1_000_000_000)  # 1 SOL
    bc.set_balance(bob_pk, 500_000_000)      # 0.5 SOL
    bc.set_balance(charlie_pk, 250_000_000)  # 0.25 SOL
    
    print(f"\nInitial balances set")
    bc.display_state()
    
    # Create transactions
    print(f"\nCreating transactions...")
    
    # Transaction 1: Alice transfers to Bob
    recent_blockhash = bc.get_recent_blockhash()
    
    builder = TransactionBuilder(alice_pk, recent_blockhash)
    transfer_instruction = SystemProgram.transfer(alice_pk, bob_pk, 100_000_000)  # 0.1 SOL
    builder.add_instruction(transfer_instruction)
    
    message1 = builder.build()
    tx1 = sign_transaction(message1, [alice_sk])
    
    # Transaction 2: Bob transfers to Charlie
    builder2 = TransactionBuilder(bob_pk, recent_blockhash)
    transfer_instruction2 = SystemProgram.transfer(bob_pk, charlie_pk, 50_000_000)  # 0.05 SOL
    builder2.add_instruction(transfer_instruction2)
    
    message2 = builder2.build()
    tx2 = sign_transaction(message2, [bob_sk])
    
    # Add transactions to mempool
    success1 = bc.add_transaction(tx1)
    success2 = bc.add_transaction(tx2)
    
    print(f"Transaction 1 added: {success1}")
    print(f"Transaction 2 added: {success2}")
    
    # Create block
    print(f"\nCreating block...")
    bc.create_block()
    
    bc.display_state()
    
    # Verify transactions
    print(f"\nTransaction verification:")
    print(f"TX1 valid signatures: {tx1.verify_signatures()}")
    print(f"TX2 valid signatures: {tx2.verify_signatures()}")
    print(f"TX1 fee: {tx1.calculate_fee():,} lamports")
    print(f"TX2 fee: {tx2.calculate_fee():,} lamports")
```

## Key Updates Based on Official Documentation

### 1. **Accurate Transaction Structure**
Now matches [Solana's transaction format](https://solana.com/docs/core/transactions):
- Multiple instructions per transaction
- Explicit account lists with access metadata
- Proper signature verification
- Recent blockhash for replay protection

### 2. **Instruction-Based Execution**
Following [Solana's instruction model](https://solana.com/docs/core/transactions):
- Instructions specify program, accounts, and data
- Account access patterns declared upfront
- Parallel execution capabilities

### 3. **Transaction Fees**
Implemented per [Solana's fee structure](https://solana.com/docs/core/fees):
- Base fee per signature (5,000 lamports)
- Compute budget for execution costs
- Priority fee mechanism

### 4. **Enhanced Mempool**
- Fee-based prioritization
- Account conflict detection
- Parallel transaction selection

## Learning Outcomes

You now understand Solana's transaction architecture:

1. **Instruction Model**: How multiple operations compose atomically
2. **Account Access**: Why explicit account lists enable parallel execution
3. **Fee Structure**: How Solana prices transaction execution
4. **Conflict Detection**: How account access patterns prevent race conditions

## Next Steps

In [Stage 4](stage-04.md), we'll add **Proof of Stake validators** who will process these transactions and create blocks using the leader selection mechanism.

---

**Key Insight**: Solana's instruction-based transaction model isn't just different—it's what enables the runtime to analyze dependencies and execute non-conflicting transactions in parallel, achieving massive throughput.

**Ready for Stage 4?** → [Validators, Staking, and Leader Selection](stage-04.md) 