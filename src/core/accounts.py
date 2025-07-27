"""
Solana Account Model Implementation

This module implements Solana's revolutionary account-based architecture where:
- All data is stored in accounts (think of it as a global database table)
- Programs (smart contracts) are stateless and stored separately from data
- Each account has an owner program that controls modifications
- Rent mechanism ensures storage cost sustainability

Based on: https://solana.com/docs/core/accounts
"""

from dataclasses import dataclass
from typing import Optional
import hashlib


@dataclass
class SolanaAccount:
    """
    Solana account structure matching the official specification.
    
    This is the foundational data structure - every piece of information
    on Solana lives in an account following this exact format.
    """
    lamports: int           # Balance in lamports (1 SOL = 1_000_000_000 lamports)
    data: bytes            # Account data (up to 10 MiB)
    owner: str             # Program ID that owns this account  
    executable: bool       # Whether this account contains executable code
    rent_epoch: int = 0    # Legacy field (always 0 in modern Solana)
    
    def __post_init__(self):
        """Validate account invariants."""
        if self.lamports < 0:
            raise ValueError("Lamports cannot be negative")
        if len(self.data) > 10 * 1024 * 1024:  # 10 MiB limit
            raise ValueError("Account data exceeds 10 MiB limit")
    
    @property
    def sol_balance(self) -> float:
        """Convert lamports to SOL for human-readable display."""
        return self.lamports / 1_000_000_000
    
    def is_rent_exempt(self, rent_per_byte_year: int = 1000) -> bool:
        """
        Check if account has enough lamports to be rent-exempt.
        
        Rent-exempt accounts don't get deleted and don't pay ongoing rent.
        The threshold is roughly 2 years of rent paid upfront.
        """
        required_balance = len(self.data) * rent_per_byte_year * 2
        return self.lamports >= required_balance
    
    def data_size(self) -> int:
        """Get size of account data in bytes."""
        return len(self.data)
    
    def copy(self) -> 'SolanaAccount':
        """Create a deep copy of this account."""
        return SolanaAccount(
            lamports=self.lamports,
            data=self.data.copy(),
            owner=self.owner,
            executable=self.executable,
            rent_epoch=self.rent_epoch
        )


@dataclass
class AccountInfo:
    """
    Account information for program execution.
    
    This is what programs see when they access accounts - it includes
    the account data plus metadata about how the account is being accessed.
    """
    key: str                # Account public key
    lamports: int          # Current lamport balance
    data: bytes            # Mutable account data
    owner: str             # Program that owns this account
    executable: bool       # Whether account is a program
    is_signer: bool        # Must have signed the transaction
    is_writable: bool      # Can be modified by the instruction
    
    def transfer_lamports_to(self, other: 'AccountInfo', amount: int) -> None:
        """Transfer lamports between accounts."""
        if amount < 0:
            raise ValueError("Cannot transfer negative amount")
        if self.lamports < amount:
            raise ValueError(f"Insufficient funds: {self.lamports} < {amount}")
        
        self.lamports -= amount
        other.lamports += amount


@dataclass
class AccountMeta:
    """
    Account metadata for instruction building.
    
    This tells the runtime how an instruction wants to access each account.
    Declaring access patterns upfront is what enables parallel execution.
    """
    pubkey: str          # Account public key
    is_signer: bool      # Must sign transaction
    is_writable: bool    # Can be modified
    
    def __str__(self) -> str:
        """Human-readable representation."""
        flags = []
        if self.is_signer:
            flags.append("signer")
        if self.is_writable:
            flags.append("writable")
        flag_str = f"({', '.join(flags)})" if flags else "(readonly)"
        return f"{self.pubkey[:8]}...{flag_str}"


class AccountRegistry:
    """
    Registry for managing all accounts in the blockchain.
    
    This is like Solana's global account database - a single source
    of truth for all account state.
    """
    
    def __init__(self):
        self._accounts: dict[str, SolanaAccount] = {}
        self._account_locks: dict[str, set[str]] = {}  # account -> set of tx hashes
    
    def create_account(self, pubkey: str, lamports: int = 0, 
                      space: int = 0, owner: str = None) -> SolanaAccount:
        """Create a new account with proper validation."""
        if pubkey in self._accounts:
            raise ValueError(f"Account {pubkey} already exists")
        
        # Default owner is System Program
        if owner is None:
            owner = "11111111111111111111111111111111"  # System Program ID
        
        account = SolanaAccount(
            lamports=lamports,
            data=bytes(space),
            owner=owner,
            executable=False
        )
        
        self._accounts[pubkey] = account
        return account
    
    def get_account(self, pubkey: str) -> Optional[SolanaAccount]:
        """Get account by public key."""
        return self._accounts.get(pubkey)
    
    def set_account(self, pubkey: str, account: SolanaAccount) -> None:
        """Set account (used for updates after instruction execution)."""
        self._accounts[pubkey] = account
    
    def account_exists(self, pubkey: str) -> bool:
        """Check if account exists."""
        return pubkey in self._accounts
    
    def get_account_balance(self, pubkey: str) -> int:
        """Get account balance in lamports."""
        account = self.get_account(pubkey)
        return account.lamports if account else 0
    
    def transfer_lamports(self, from_pubkey: str, to_pubkey: str, amount: int) -> bool:
        """Transfer lamports between accounts."""
        from_account = self.get_account(from_pubkey)
        to_account = self.get_account(to_pubkey)
        
        if not from_account or not to_account:
            return False
        
        if from_account.lamports < amount:
            return False
        
        # Create updated accounts
        updated_from = from_account.copy()
        updated_to = to_account.copy()
        
        updated_from.lamports -= amount
        updated_to.lamports += amount
        
        # Update registry
        self.set_account(from_pubkey, updated_from)
        self.set_account(to_pubkey, updated_to)
        
        return True
    
    def get_accounts_by_owner(self, owner: str) -> dict[str, SolanaAccount]:
        """Get all accounts owned by a specific program."""
        return {
            pubkey: account 
            for pubkey, account in self._accounts.items() 
            if account.owner == owner
        }
    
    def get_program_accounts(self) -> dict[str, SolanaAccount]:
        """Get all executable accounts (programs)."""
        return {
            pubkey: account 
            for pubkey, account in self._accounts.items() 
            if account.executable
        }
    
    def total_lamports(self) -> int:
        """Get total lamports in all accounts."""
        return sum(account.lamports for account in self._accounts.values())
    
    def account_count(self) -> int:
        """Get total number of accounts."""
        return len(self._accounts)
    
    def largest_accounts(self, limit: int = 10) -> list[tuple[str, SolanaAccount]]:
        """Get accounts with most data, useful for debugging."""
        return sorted(
            self._accounts.items(),
            key=lambda x: len(x[1].data),
            reverse=True
        )[:limit]
    
    def __len__(self) -> int:
        """Number of accounts in registry."""
        return len(self._accounts)
    
    def __contains__(self, pubkey: str) -> bool:
        """Check if account exists using 'in' operator."""
        return pubkey in self._accounts
    
    def __getitem__(self, pubkey: str) -> SolanaAccount:
        """Get account using bracket notation."""
        account = self.get_account(pubkey)
        if account is None:
            raise KeyError(f"Account {pubkey} not found")
        return account 