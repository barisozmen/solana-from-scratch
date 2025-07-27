# Stage 7: Smart Contracts - Solana's Account Model and Programs

> *"On Solana, all data is stored in what are called 'accounts.' You can think of data on Solana as a public database with a single 'Accounts' table."* - [Solana Documentation](https://solana.com/docs/core/accounts)

## Objectives

This stage implements Solana's unique account model and program architecture, which differs significantly from Ethereum's approach:

- Understand Solana's account-based data storage model
- Implement the separation between programs (code) and accounts (data)
- Create Program Derived Addresses (PDAs) for deterministic account generation
- Build Cross Program Invocation (CPI) for program composability
- Handle rent-exempt accounts and proper account lifecycle management

## Why Solana's Model is Different

### Traditional Smart Contract Limitations
Ethereum-style contracts have:
- **Code and data together**: Contract state stored with code
- **Account creation complexity**: Manual state management
- **Limited composability**: Difficult cross-contract calls
- **Sequential execution**: One transaction at a time

### Solana's Revolutionary Approach
Based on the [official Solana account model](https://solana.com/docs/core/accounts):
- **Separation of concerns**: Programs (code) separate from accounts (data)
- **Stateless programs**: Code is immutable, state is in separate accounts
- **Rent mechanism**: Accounts pay for storage proportional to data size
- **Parallel execution**: Non-conflicting programs run simultaneously
- **Native composability**: Cross Program Invocation (CPI) enables seamless interaction

## Key Concepts from Solana Documentation

### 1. Account Structure
Every Solana account has exactly these fields:
```rust
pub struct Account {
    pub lamports: u64,      // Balance in lamports (1 SOL = 1B lamports)
    pub data: Vec<u8>,      // Arbitrary data storage
    pub owner: Pubkey,      // Program that owns this account
    pub executable: bool,   // Whether this is a program
    pub rent_epoch: Epoch,  // Legacy field (no longer used)
}
```

### 2. Account Types
- **Program Accounts**: Store executable code (`executable: true`)
- **Data Accounts**: Store program state (`executable: false`)
- **Sysvar Accounts**: Special accounts storing network state

### 3. Program Derived Addresses (PDAs)
Deterministic addresses derived from:
- Program ID
- Seeds (arbitrary data)
- Bump seed (ensures address is off the Ed25519 curve)

### 4. Rent Mechanism
- Accounts must maintain minimum balance proportional to data size
- **Rent-exempt threshold**: ~2 years of rent paid upfront
- Accounts below threshold are deleted during rent collection

## Implementation

### Updated Account Structure

```python
@dataclass
class SolanaAccount:
    """Accurate Solana account structure per official documentation."""
    lamports: int           # Balance in lamports (1 SOL = 1_000_000_000 lamports)
    data: bytes            # Account data storage
    owner: str             # Program ID that owns this account  
    executable: bool       # Whether this account contains executable code
    rent_epoch: int        # Legacy field, always 0 in modern Solana
    
    def __post_init__(self):
        if self.lamports < 0:
            raise ValueError("Lamports cannot be negative")
    
    def is_rent_exempt(self, rent_per_byte_year: int = 1000) -> bool:
        """Check if account has enough lamports to be rent-exempt."""
        # Simplified rent calculation: ~2 years of rent
        required_balance = len(self.data) * rent_per_byte_year * 2
        return self.lamports >= required_balance
    
    def data_size(self) -> int:
        """Get size of account data in bytes."""
        return len(self.data)

    @property
    def sol_balance(self) -> float:
        """Convert lamports to SOL."""
        return self.lamports / 1_000_000_000

@dataclass
class AccountInfo:
    """Account information for program execution."""
    key: str
    lamports: int
    data: bytes
    owner: str
    executable: bool
    is_signer: bool
    is_writable: bool

@dataclass
class AccountMeta:
    """Account metadata for instructions."""
    pubkey: str
    is_signer: bool
    is_writable: bool

@dataclass
class Instruction:
    """Solana instruction structure."""
    program_id: str
    accounts: List[AccountMeta]
    data: bytes

class PDAGenerator:
    """Generate Program Derived Addresses as per Solana specification."""
    
    MAX_SEED_LENGTH = 32
    PDA_MARKER = b"ProgramDerivedAddress"
    
    @staticmethod
    def find_program_address(seeds: List[bytes], program_id: str) -> Tuple[str, int]:
        """Find a valid program address and bump seed."""
        for seed in seeds:
            if len(seed) > PDAGenerator.MAX_SEED_LENGTH:
                raise ValueError(f"Seed too long: {len(seed)} > {PDAGenerator.MAX_SEED_LENGTH}")
        
        for bump in range(255, -1, -1):
            try:
                address = PDAGenerator._create_program_address(seeds + [bytes([bump])], program_id)
                return address, bump
            except ValueError:
                continue
        
        raise ValueError("Unable to find a valid program address")
    
    @staticmethod
    def _create_program_address(seeds: List[bytes], program_id: str) -> str:
        """Create program address from seeds and program ID."""
        total_len = sum(len(seed) for seed in seeds)
        if total_len > PDAGenerator.MAX_SEED_LENGTH:
            raise ValueError("Seeds too long")
        
        hash_input = b"".join(seeds) + bytes.fromhex(program_id) + PDAGenerator.PDA_MARKER
        hash_result = hashlib.sha256(hash_input).digest()
        
        # Simplified on-curve check
        if hash_result[-1] < 128:
            raise ValueError("Address is on curve")
        
        return hash_result.hex()

class ProgramError(Exception):
    pass

class Program(ABC):
    """Base class for all Solana programs."""
    
    @abstractmethod
    def process_instruction(self, instruction_data: bytes, accounts: List[AccountInfo]) -> None:
        pass
    
    def _transfer_lamports(self, from_account: AccountInfo, to_account: AccountInfo, amount: int) -> None:
        if from_account.lamports < amount:
            raise ProgramError(f"Insufficient funds: {from_account.lamports} < {amount}")
        from_account.lamports -= amount
        to_account.lamports += amount
    
    def _check_writable(self, account: AccountInfo) -> None:
        if not account.is_writable:
            raise ProgramError(f"Account {account.key} is not writable")
    
    def _check_signer(self, account: AccountInfo) -> None:
        if not account.is_signer:
            raise ProgramError(f"Account {account.key} must be signer")

class SystemProgram(Program):
    """Solana's built-in System Program."""
    
    PROGRAM_ID = "11111111111111111111111111111111"
    
    CREATE_ACCOUNT = 0
    TRANSFER = 2
    ALLOCATE = 8
    ASSIGN = 1
    
    def process_instruction(self, instruction_data: bytes, accounts: List[AccountInfo]) -> None:
        if len(instruction_data) == 0:
            raise ProgramError("Empty instruction data")
        
        instruction_type = instruction_data[0]
        
        if instruction_type == self.CREATE_ACCOUNT:
            self._create_account(instruction_data[1:], accounts)
        elif instruction_type == self.TRANSFER:
            self._transfer(instruction_data[1:], accounts)
        elif instruction_type == self.ALLOCATE:
            self._allocate(instruction_data[1:], accounts)
        elif instruction_type == self.ASSIGN:
            self._assign(instruction_data[1:], accounts)
        else:
            raise ProgramError(f"Unknown system instruction: {instruction_type}")
    
    def _create_account(self, data: bytes, accounts: List[AccountInfo]) -> None:
        if len(accounts) < 2:
            raise ProgramError("Create account requires 2 accounts")
        
        payer = accounts[0]
        new_account = accounts[1]
        
        self._check_signer(payer)
        self._check_signer(new_account)
        self._check_writable(payer)
        self._check_writable(new_account)
        
        if len(data) < 52:
            raise ProgramError("Invalid create account data")
        
        lamports = int.from_bytes(data[0:8], 'little')
        space = int.from_bytes(data[8:16], 'little')
        owner = data[16:48].hex()
        
        if space > 10 * 1024 * 1024:  # 10 MiB limit
            raise ProgramError("Account data too large")
        
        if new_account.lamports > 0 or len(new_account.data) > 0:
            raise ProgramError("Account already in use")
        
        self._transfer_lamports(payer, new_account, lamports)
        new_account.data = bytes(space)
        new_account.owner = owner
        
        print(f"Created account {new_account.key[:8]}... with {lamports} lamports, {space} bytes")
    
    def _transfer(self, data: bytes, accounts: List[AccountInfo]) -> None:
        if len(accounts) < 2:
            raise ProgramError("Transfer requires 2 accounts")
        
        from_account = accounts[0]
        to_account = accounts[1]
        
        self._check_signer(from_account)
        self._check_writable(from_account)
        self._check_writable(to_account)
        
        if len(data) < 8:
            raise ProgramError("Invalid transfer data")
        
        amount = int.from_bytes(data[0:8], 'little')
        self._transfer_lamports(from_account, to_account, amount)
        print(f"Transferred {amount} lamports from {from_account.key[:8]}... to {to_account.key[:8]}...")
    
    def _allocate(self, data: bytes, accounts: List[AccountInfo]) -> None:
        if len(accounts) < 1:
            raise ProgramError("Allocate requires 1 account")
        
        account = accounts[0]
        self._check_signer(account)
        self._check_writable(account)
        
        if len(data) < 8:
            raise ProgramError("Invalid allocate data")
        
        space = int.from_bytes(data[0:8], 'little')
        if space > 10 * 1024 * 1024:
            raise ProgramError("Account data too large")
        
        account.data = bytes(space)
        print(f"Allocated {space} bytes for account {account.key[:8]}...")
    
    def _assign(self, data: bytes, accounts: List[AccountInfo]) -> None:
        if len(accounts) < 1:
            raise ProgramError("Assign requires 1 account")
        
        account = accounts[0]
        self._check_signer(account)
        self._check_writable(account)
        
        if len(data) < 32:
            raise ProgramError("Invalid assign data")
        
        new_owner = data[0:32].hex()
        account.owner = new_owner
        print(f"Assigned account {account.key[:8]}... to program {new_owner[:8]}...")

class SplTokenProgram(Program):
    """Simplified SPL Token Program demonstrating Solana's account model."""
    
    PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    
    MINT_DISCRIMINATOR = b"\x00\x00\x00\x00\x00\x00\x00\x01"
    TOKEN_ACCOUNT_DISCRIMINATOR = b"\x00\x00\x00\x00\x00\x00\x00\x02"
    
    INITIALIZE_MINT = 0
    INITIALIZE_ACCOUNT = 1
    MINT_TO = 7
    TRANSFER = 3
    
    def process_instruction(self, instruction_data: bytes, accounts: List[AccountInfo]) -> None:
        if len(instruction_data) == 0:
            raise ProgramError("Empty instruction data")
        
        instruction_type = instruction_data[0]
        
        if instruction_type == self.INITIALIZE_MINT:
            self._initialize_mint(instruction_data[1:], accounts)
        elif instruction_type == self.INITIALIZE_ACCOUNT:
            self._initialize_account(instruction_data[1:], accounts)
        elif instruction_type == self.MINT_TO:
            self._mint_to(instruction_data[1:], accounts)
        elif instruction_type == self.TRANSFER:
            self._transfer_tokens(instruction_data[1:], accounts)
        else:
            raise ProgramError(f"Unknown token instruction: {instruction_type}")
    
    def _initialize_mint(self, data: bytes, accounts: List[AccountInfo]) -> None:
        if len(accounts) < 1:
            raise ProgramError("Initialize mint requires mint account")
        
        mint = accounts[0]
        self._check_writable(mint)
        
        if mint.owner != self.PROGRAM_ID:
            raise ProgramError("Mint account not owned by token program")
        
        if len(mint.data) < 82:
            raise ProgramError("Mint account too small")
        
        if len(data) < 33:
            raise ProgramError("Invalid mint data")
        
        decimals = data[0]
        mint_authority = data[1:33].hex()
        
        # Initialize mint data
        mint_data = bytearray(82)
        mint_data[0:8] = self.MINT_DISCRIMINATOR
        mint_data[8] = decimals
        mint_data[9:41] = bytes.fromhex(mint_authority)
        mint_data[49] = 1  # is_initialized
        
        mint.data = bytes(mint_data)
        print(f"Initialized mint {mint.key[:8]}... with {decimals} decimals")
    
    def _initialize_account(self, data: bytes, accounts: List[AccountInfo]) -> None:
        if len(accounts) < 3:
            raise ProgramError("Initialize account requires 3 accounts")
        
        token_account = accounts[0]
        mint = accounts[1]
        owner = accounts[2]
        
        self._check_writable(token_account)
        
        if token_account.owner != self.PROGRAM_ID:
            raise ProgramError("Token account not owned by token program")
        
        if len(token_account.data) < 165:
            raise ProgramError("Token account too small")
        
        # Initialize token account data
        account_data = bytearray(165)
        account_data[0:8] = self.TOKEN_ACCOUNT_DISCRIMINATOR
        account_data[8:40] = bytes.fromhex(mint.key)
        account_data[40:72] = bytes.fromhex(owner.key)
        account_data[80] = 1  # is_initialized
        
        token_account.data = bytes(account_data)
        print(f"Initialized token account {token_account.key[:8]}...")
    
    def _mint_to(self, data: bytes, accounts: List[AccountInfo]) -> None:
        if len(accounts) < 3:
            raise ProgramError("Mint to requires 3 accounts")
        
        mint = accounts[0]
        token_account = accounts[1]
        authority = accounts[2]
        
        self._check_writable(mint)
        self._check_writable(token_account)
        self._check_signer(authority)
        
        if len(data) < 8:
            raise ProgramError("Invalid mint amount")
        
        amount = int.from_bytes(data[0:8], 'little')
        
        # Verify authority
        mint_data = bytearray(mint.data)
        mint_authority = mint_data[9:41].hex()
        if authority.key != mint_authority:
            raise ProgramError("Invalid mint authority")
        
        # Update mint supply
        current_supply = int.from_bytes(mint_data[41:49], 'little')
        new_supply = current_supply + amount
        mint_data[41:49] = new_supply.to_bytes(8, 'little')
        mint.data = bytes(mint_data)
        
        # Update token account balance
        account_data = bytearray(token_account.data)
        current_balance = int.from_bytes(account_data[72:80], 'little')
        new_balance = current_balance + amount
        account_data[72:80] = new_balance.to_bytes(8, 'little')
        token_account.data = bytes(account_data)
        
        print(f"Minted {amount} tokens to {token_account.key[:8]}...")
    
    def _transfer_tokens(self, data: bytes, accounts: List[AccountInfo]) -> None:
        if len(accounts) < 3:
            raise ProgramError("Transfer requires 3 accounts")
        
        from_account = accounts[0]
        to_account = accounts[1]
        owner = accounts[2]
        
        self._check_writable(from_account)
        self._check_writable(to_account)
        self._check_signer(owner)
        
        if len(data) < 8:
            raise ProgramError("Invalid transfer amount")
        
        amount = int.from_bytes(data[0:8], 'little')
        
        # Verify owner
        from_data = bytearray(from_account.data)
        account_owner = from_data[40:72].hex()
        if owner.key != account_owner:
            raise ProgramError("Invalid token account owner")
        
        # Update from account
        from_balance = int.from_bytes(from_data[72:80], 'little')
        if from_balance < amount:
            raise ProgramError(f"Insufficient token balance: {from_balance} < {amount}")
        
        new_from_balance = from_balance - amount
        from_data[72:80] = new_from_balance.to_bytes(8, 'little')
        from_account.data = bytes(from_data)
        
        # Update to account
        to_data = bytearray(to_account.data)
        to_balance = int.from_bytes(to_data[72:80], 'little')
        new_to_balance = to_balance + amount
        to_data[72:80] = new_to_balance.to_bytes(8, 'little')
        to_account.data = bytes(to_data)
        
        print(f"Transferred {amount} tokens from {from_account.key[:8]}... to {to_account.key[:8]}...")

class CrossProgramInvocation:
    """Enable programs to call other programs (CPI)."""
    
    def __init__(self, runtime: 'SolanaRuntime'):
        self.runtime = runtime
        self.call_stack: List[str] = []
        self.max_call_depth = 4  # Solana's limit
    
    def invoke(self, instruction: Instruction, signers: List[str]) -> bool:
        if len(self.call_stack) >= self.max_call_depth:
            raise ProgramError(f"Call stack overflow: max depth {self.max_call_depth}")
        
        self.call_stack.append(instruction.program_id)
        
        try:
            return self.runtime.execute_instruction(instruction, signers)
        finally:
            self.call_stack.pop()
    
    def invoke_signed(self, instruction: Instruction, signer_seeds: List[List[bytes]]) -> bool:
        if not self.call_stack:
            raise ProgramError("invoke_signed can only be called from within a program")
        
        current_program = self.call_stack[-1]
        
        # Derive signers from seeds
        derived_signers = []
        for seeds in signer_seeds:
            try:
                address, _ = PDAGenerator.find_program_address(seeds, current_program)
                derived_signers.append(address)
            except ValueError:
                raise ProgramError("Invalid signer seeds")
        
        return self.invoke(instruction, derived_signers)

class SolanaRuntime:
    """Enhanced runtime with Solana-specific features."""
    
    def __init__(self, accounts: Dict[str, SolanaAccount]):
        self.accounts = accounts
        self.programs: Dict[str, Program] = {}
        self.cpi = CrossProgramInvocation(self)
        
        # Register built-in programs
        self.programs[SystemProgram.PROGRAM_ID] = SystemProgram()
        self.programs[SplTokenProgram.PROGRAM_ID] = SplTokenProgram()
    
    def register_program(self, program_id: str, program: Program) -> None:
        self.programs[program_id] = program
    
    def execute_instruction(self, instruction: Instruction, signers: List[str]) -> bool:
        try:
            program = self.programs.get(instruction.program_id)
            if not program:
                raise ProgramError(f"Program {instruction.program_id} not found")
            
            # Prepare account infos
            account_infos = []
            for account_meta in instruction.accounts:
                account = self.accounts.get(account_meta.pubkey)
                if not account:
                    raise ProgramError(f"Account {account_meta.pubkey} not found")
                
                if account_meta.is_signer and account_meta.pubkey not in signers:
                    raise ProgramError(f"Account {account_meta.pubkey} must be signer")
                
                account_info = AccountInfo(
                    key=account_meta.pubkey,
                    lamports=account.lamports,
                    data=account.data.copy(),
                    owner=account.owner,
                    executable=account.executable,
                    is_signer=account_meta.is_signer,
                    is_writable=account_meta.is_writable
                )
                account_infos.append(account_info)
            
            # Execute program
            program.process_instruction(instruction.data, account_infos)
            
            # Apply changes back to accounts
            for i, account_meta in enumerate(instruction.accounts):
                if account_meta.is_writable:
                    account_info = account_infos[i]
                    account = self.accounts[account_meta.pubkey]
                    account.lamports = account_info.lamports
                    account.data = account_info.data
                    account.owner = account_info.owner
            
            return True
            
        except ProgramError as e:
            print(f"Program execution failed: {e}")
            return False

class SolanaBlockchain(TowerBFTBlockchain):
    """Enhanced blockchain with accurate Solana account model."""
    
    def __init__(self):
        super().__init__()
        self.solana_accounts: Dict[str, SolanaAccount] = {}
        self.runtime = SolanaRuntime(self.solana_accounts)
        
        # Initialize system accounts
        self._initialize_system_accounts()
    
    def _initialize_system_accounts(self):
        """Create initial system accounts."""
        # System program account
        system_account = SolanaAccount(
            lamports=0,
            data=b"",
            owner=SystemProgram.PROGRAM_ID,
            executable=True
        )
        self.solana_accounts[SystemProgram.PROGRAM_ID] = system_account
        
        # Token program account
        token_account = SolanaAccount(
            lamports=0,
            data=b"",
            owner=SystemProgram.PROGRAM_ID,
            executable=True
        )
        self.solana_accounts[SplTokenProgram.PROGRAM_ID] = token_account
    
    def create_account(self, pubkey: str, lamports: int = 0, space: int = 0, owner: str = None) -> None:
        """Create a new account with proper Solana structure."""
        if pubkey not in self.solana_accounts:
            account = SolanaAccount(
                lamports=lamports,
                data=bytes(space),
                owner=owner or SystemProgram.PROGRAM_ID,
                executable=False
            )
            self.solana_accounts[pubkey] = account
    
    def get_account(self, pubkey: str) -> Optional[SolanaAccount]:
        """Get account by public key."""
        return self.solana_accounts.get(pubkey)
    
    def execute_instruction(self, instruction: Instruction, signers: List[str]) -> bool:
        """Execute a single instruction."""
        return self.runtime.execute_instruction(instruction, signers)
    
    def display_solana_state(self):
        """Display Solana-specific state."""
        print(f"\n{'='*60}")
        print("SOLANA BLOCKCHAIN STATE")
        print(f"{'='*60}")
        
        print(f"Total accounts: {len(self.solana_accounts)}")
        print(f"Programs registered: {len(self.runtime.programs)}")
        
        # Show account breakdown
        executable_count = sum(1 for acc in self.solana_accounts.values() if acc.executable)
        data_count = len(self.solana_accounts) - executable_count
        
        print(f"Program accounts: {executable_count}")
        print(f"Data accounts: {data_count}")
        
        # Show largest accounts
        print(f"\nLargest accounts by data size:")
        sorted_accounts = sorted(
            self.solana_accounts.items(),
            key=lambda x: len(x[1].data),
            reverse=True
        )
        
        for pubkey, account in sorted_accounts[:5]:
            account_type = "Program" if account.executable else "Data"
            print(f"  {pubkey[:8]}...: {account_type}, {len(account.data)} bytes, {account.sol_balance:.6f} SOL")

# Demo functions
def demo_pda_creation():
    """Demonstrate Program Derived Address creation."""
    print("=== PDA Creation Demo ===")
    
    # Create PDA for associated token account
    owner = "11111111111111111111111111111112"  # Example owner
    mint = "11111111111111111111111111111113"   # Example mint
    token_program = SplTokenProgram.PROGRAM_ID
    
    try:
        pda, bump = PDAGenerator.find_program_address([
            bytes.fromhex(owner),
            bytes.fromhex(token_program),
            bytes.fromhex(mint)
        ], token_program)
        
        print(f"Owner: {owner}")
        print(f"Mint: {mint}")
        print(f"Token Program: {token_program}")
        print(f"Generated PDA: {pda}")
        print(f"Bump seed: {bump}")
        
    except ValueError as e:
        print(f"PDA generation failed: {e}")

def demo_token_program():
    """Demonstrate SPL Token Program functionality."""
    print("\n=== Token Program Demo ===")
    
    # Create blockchain
    bc = SolanaBlockchain()
    
    # Generate accounts
    authority_sk = SigningKey.generate(curve=SECP256k1)
    authority_pk = authority_sk.verifying_key.to_string().hex()
    
    mint_sk = SigningKey.generate(curve=SECP256k1)
    mint_pk = mint_sk.verifying_key.to_string().hex()
    
    token_account_sk = SigningKey.generate(curve=SECP256k1)
    token_account_pk = token_account_sk.verifying_key.to_string().hex()
    
    # Create accounts
    bc.create_account(authority_pk, lamports=1000000000)  # 1 SOL
    bc.create_account(mint_pk, space=82, owner=SplTokenProgram.PROGRAM_ID)
    bc.create_account(token_account_pk, space=165, owner=SplTokenProgram.PROGRAM_ID)
    
    print(f"Authority: {authority_pk[:8]}...")
    print(f"Mint: {mint_pk[:8]}...")
    print(f"Token Account: {token_account_pk[:8]}...")
    
    # Initialize mint
    init_mint_instruction = Instruction(
        program_id=SplTokenProgram.PROGRAM_ID,
        accounts=[
            AccountMeta(mint_pk, False, True)
        ],
        data=bytes([SplTokenProgram.INITIALIZE_MINT]) + bytes([6]) + bytes.fromhex(authority_pk)
    )
    
    success = bc.execute_instruction(init_mint_instruction, [])
    print(f"Initialize mint: {'Success' if success else 'Failed'}")
    
    # Initialize token account
    init_account_instruction = Instruction(
        program_id=SplTokenProgram.PROGRAM_ID,
        accounts=[
            AccountMeta(token_account_pk, False, True),
            AccountMeta(mint_pk, False, False),
            AccountMeta(authority_pk, False, False)
        ],
        data=bytes([SplTokenProgram.INITIALIZE_ACCOUNT])
    )
    
    success = bc.execute_instruction(init_account_instruction, [])
    print(f"Initialize token account: {'Success' if success else 'Failed'}")
    
    # Mint tokens
    mint_to_instruction = Instruction(
        program_id=SplTokenProgram.PROGRAM_ID,
        accounts=[
            AccountMeta(mint_pk, False, True),
            AccountMeta(token_account_pk, False, True),
            AccountMeta(authority_pk, True, False)
        ],
        data=bytes([SplTokenProgram.MINT_TO]) + (1000000).to_bytes(8, 'little')
    )
    
    success = bc.execute_instruction(mint_to_instruction, [authority_pk])
    print(f"Mint tokens: {'Success' if success else 'Failed'}")
    
    # Display final state
    bc.display_solana_state()

if __name__ == "__main__":
    print("Solana Account Model and Programs Demo")
    print("=" * 50)
    
    demo_pda_creation()
    demo_token_program()
```

## Key Updates Based on Official Documentation

### 1. **Accurate Account Structure**
Now matches [Solana's exact account fields](https://solana.com/docs/core/accounts):
- `lamports`: Balance in smallest unit (1 SOL = 1B lamports)
- `data`: Up to 10 MiB of arbitrary data
- `owner`: Program that controls the account
- `executable`: Whether account contains program code
- `rent_epoch`: Legacy field (always 0)

### 2. **Program Derived Addresses (PDAs)**
Implemented per [official PDA specification](https://solana.com/docs/core/pda):
- Deterministic address generation from seeds + program ID
- Bump seed to ensure address is off Ed25519 curve
- Used for creating accounts without private keys

### 3. **Cross Program Invocation (CPI)**
Following [Solana's CPI model](https://solana.com/docs/core/cpi):
- Programs can call other programs
- `invoke()` and `invoke_signed()` functions
- Call stack depth limits (4 levels)
- PDA signing for derived accounts

### 4. **Rent-Exempt Accounts**
Based on [Solana's rent mechanism](https://solana.com/docs/core/accounts#rent):
- Accounts must maintain minimum balance proportional to data size
- Rent-exempt threshold calculation
- Account lifecycle management

### 5. **System Program**
Accurate implementation of [Solana's System Program](https://solana.com/docs/core/programs):
- Account creation with proper ownership transfer
- Lamport transfers between accounts
- Space allocation and program assignment
- Standard instruction set

## Learning Outcomes

You now understand Solana's revolutionary architecture:

1. **Account Model**: How data and code separation enables parallel execution
2. **Program Ownership**: Why accounts have owners and how it ensures security
3. **PDAs**: How to create deterministic addresses for program-controlled accounts
4. **CPI**: How programs compose together for complex applications
5. **Rent System**: How Solana manages storage costs and account lifecycle

## Next Steps

In [Stage 8](stage-08.md), we'll leverage this proper Solana account model to implement **Sealevel-style parallel execution**, where our dependency analysis can now properly identify account conflicts using the owner/writable patterns that make Solana so fast.

---

**Key Insight**: Solana's account model isn't just different—it's what enables the 50,000+ TPS throughput by allowing parallel execution of non-conflicting transactions across separate account spaces.

**Ready for Stage 8?** → [Parallel Execution with Sealevel](stage-08.md) 