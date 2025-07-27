"""
Solana Transaction and Instruction Model

This implements Solana's revolutionary transaction structure where:
- Transactions contain multiple instructions that execute atomically
- All account access is declared upfront, enabling parallel execution
- Instructions specify program, accounts, and data explicitly
- Fee structure includes base fees + compute budget

Based on: https://solana.com/docs/core/transactions
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict
from ecdsa import SigningKey, VerifyingKey, SECP256k1

from .accounts import AccountMeta


@dataclass
class MessageHeader:
    """
    Transaction message header with account access metadata.
    
    This tells the runtime how many accounts need to sign and
    which accounts are read-only vs writable.
    """
    num_required_signatures: int      # Number of signatures required
    num_readonly_signed_accounts: int # Read-only accounts that must sign
    num_readonly_unsigned_accounts: int # Read-only accounts (no signature)


@dataclass
class CompiledInstruction:
    """
    Instruction compiled to reference accounts by index.
    
    Instead of embedding full account keys, we reference them by their
    position in the transaction's account array. This saves space and
    enables efficient processing.
    """
    program_id_index: int           # Index into account_keys for program
    accounts: List[int]            # Indices into account_keys
    data: bytes                    # Program-specific instruction data
    
    def __str__(self) -> str:
        return f"Instruction(program_id_index={self.program_id_index}, accounts={self.accounts}, data_len={len(self.data)})"


@dataclass
class TransactionMessage:
    """
    The core transaction message structure matching Solana's specification.
    
    This contains all the information needed to execute a transaction,
    organized for efficient parallel processing.
    """
    header: MessageHeader
    account_keys: List[str]        # All account public keys referenced
    recent_blockhash: str          # Recent blockhash for replay protection
    instructions: List[CompiledInstruction]
    
    def serialize(self) -> bytes:
        """
        Serialize message for signing and transmission.
        
        This implements a simplified version of Solana's binary format.
        Real Solana uses bincode for more efficient serialization.
        """
        parts = []
        
        # Header
        parts.append(self.header.num_required_signatures.to_bytes(1, 'little'))
        parts.append(self.header.num_readonly_signed_accounts.to_bytes(1, 'little'))
        parts.append(self.header.num_readonly_unsigned_accounts.to_bytes(1, 'little'))
        
        # Account keys
        parts.append(len(self.account_keys).to_bytes(1, 'little'))
        for key in self.account_keys:
            parts.append(bytes.fromhex(key))
        
        # Recent blockhash
        parts.append(bytes.fromhex(self.recent_blockhash))
        
        # Instructions
        parts.append(len(self.instructions).to_bytes(1, 'little'))
        for instruction in self.instructions:
            parts.append(instruction.program_id_index.to_bytes(1, 'little'))
            parts.append(len(instruction.accounts).to_bytes(1, 'little'))
            parts.extend(acc.to_bytes(1, 'little') for acc in instruction.accounts)
            parts.append(len(instruction.data).to_bytes(2, 'little'))
            parts.append(instruction.data)
        
        return b''.join(parts)
    
    def get_account_access_pattern(self) -> Dict[str, Set[str]]:
        """
        Analyze which accounts are accessed by which instructions.
        
        Returns mapping of account -> set of instruction types
        This is used for dependency analysis in parallel execution.
        """
        access_pattern = {}
        
        for i, instruction in enumerate(self.instructions):
            program_id = self.account_keys[instruction.program_id_index]
            
            for account_index in instruction.accounts:
                if account_index < len(self.account_keys):
                    account_key = self.account_keys[account_index]
                    if account_key not in access_pattern:
                        access_pattern[account_key] = set()
                    access_pattern[account_key].add(f"instr_{i}_{program_id[:8]}")
        
        return access_pattern


@dataclass
class SolanaTransaction:
    """
    Complete Solana transaction with signatures and message.
    
    This is the final transaction structure that gets included in blocks
    and broadcasted across the network.
    """
    signatures: List[str]          # Ed25519 signatures in hex
    message: TransactionMessage
    
    def hash(self) -> str:
        """Compute deterministic transaction hash."""
        return hashlib.sha256(self.message.serialize()).hexdigest()
    
    def verify_signatures(self) -> bool:
        """
        Verify all transaction signatures.
        
        Each required signer must provide a valid Ed25519 signature
        over the serialized message.
        """
        message_data = self.message.serialize()
        
        # Check we have enough signatures
        required_sigs = self.message.header.num_required_signatures
        if len(self.signatures) < required_sigs:
            return False
        
        # Verify each required signature
        for i in range(required_sigs):
            if i >= len(self.signatures) or i >= len(self.message.account_keys):
                return False
            
            try:
                signer_pubkey = self.message.account_keys[i]
                signature = self.signatures[i]
                
                # Reconstruct public key and verify signature
                pubkey_bytes = bytes.fromhex(signer_pubkey)
                sig_bytes = bytes.fromhex(signature)
                
                vk = VerifyingKey.from_string(pubkey_bytes, curve=SECP256k1)
                if not vk.verify(sig_bytes, message_data):
                    return False
                    
            except Exception:
                return False
        
        return True
    
    def get_fee_payer(self) -> str:
        """Get the fee payer (always the first signer)."""
        if not self.message.account_keys:
            raise ValueError("Transaction has no accounts")
        return self.message.account_keys[0]
    
    def calculate_fee(self) -> int:
        """
        Calculate transaction fee in lamports.
        
        Solana's fee structure:
        - Base fee: 5,000 lamports per signature
        - Compute budget: Based on computational complexity
        """
        # Base fee: 5,000 lamports per signature
        base_fee = len(self.signatures) * 5000
        
        # Compute budget fee (simplified calculation)
        compute_units = 0
        for instruction in self.message.instructions:
            # Rough estimate: 1000 CU base + 100 per byte of instruction data
            compute_units += 1000 + len(instruction.data) * 100
        
        # Cap at 200k compute units and convert to fee
        compute_units = min(compute_units, 200_000)
        compute_fee = compute_units // 1000  # 1 lamport per 1000 CU
        
        return base_fee + compute_fee
    
    def get_writable_accounts(self) -> Set[str]:
        """
        Get all accounts that this transaction can modify.
        
        This is crucial for parallel execution - transactions that
        write to the same accounts cannot run simultaneously.
        """
        writable = set()
        header = self.message.header
        
        # Fee payer is always writable
        writable.add(self.get_fee_payer())
        
        # Determine writable accounts based on Solana's ordering
        num_required_sigs = header.num_required_signatures
        num_readonly_signed = header.num_readonly_signed_accounts
        num_readonly_unsigned = header.num_readonly_unsigned_accounts
        
        total_accounts = len(self.message.account_keys)
        
        for i, account_key in enumerate(self.message.account_keys):
            # Required signers (excluding readonly signed) are writable
            if i < num_required_sigs:
                writable.add(account_key)
            # Skip readonly signed accounts
            elif i < num_required_sigs + num_readonly_signed:
                continue
            # Writable unsigned accounts
            elif i < total_accounts - num_readonly_unsigned:
                writable.add(account_key)
            # Readonly unsigned accounts are not writable
        
        return writable
    
    def get_readonly_accounts(self) -> Set[str]:
        """Get all accounts this transaction only reads from."""
        all_accounts = set(self.message.account_keys)
        writable_accounts = self.get_writable_accounts()
        return all_accounts - writable_accounts


@dataclass
class Instruction:
    """
    High-level instruction before compilation to indices.
    
    This is the developer-friendly format for building transactions.
    It gets compiled down to CompiledInstruction for efficiency.
    """
    program_id: str                # Program to invoke
    accounts: List[AccountMeta]    # Accounts with access metadata
    data: bytes                   # Instruction data
    
    def __str__(self) -> str:
        return f"Instruction({self.program_id[:8]}..., {len(self.accounts)} accounts, {len(self.data)} bytes)"


class TransactionBuilder:
    """
    Builder for constructing Solana transactions elegantly.
    
    This handles the complex logic of ordering accounts correctly
    and compiling instructions to their efficient binary format.
    """
    
    def __init__(self, fee_payer: str, recent_blockhash: str):
        """
        Initialize transaction builder.
        
        Args:
            fee_payer: Account that pays transaction fees (must be signer)
            recent_blockhash: Recent blockhash for replay protection
        """
        self.fee_payer = fee_payer
        self.recent_blockhash = recent_blockhash
        self.instructions: List[Instruction] = []
    
    def add_instruction(self, instruction: Instruction) -> 'TransactionBuilder':
        """Add an instruction to the transaction (fluent interface)."""
        self.instructions.append(instruction)
        return self
    
    def add_instructions(self, instructions: List[Instruction]) -> 'TransactionBuilder':
        """Add multiple instructions at once."""
        self.instructions.extend(instructions)
        return self
    
    def build(self) -> TransactionMessage:
        """
        Build the final transaction message.
        
        This performs the complex task of:
        1. Collecting all unique accounts
        2. Ordering them according to Solana's rules
        3. Compiling instructions to use indices
        4. Creating the proper message header
        """
        # Collect all unique accounts
        all_accounts = {self.fee_payer}  # Fee payer is always first
        
        for instruction in self.instructions:
            all_accounts.add(instruction.program_id)
            for account in instruction.accounts:
                all_accounts.add(account.pubkey)
        
        # Categorize accounts
        signer_accounts = {self.fee_payer}  # Fee payer always signs
        writable_accounts = {self.fee_payer}  # Fee payer always writable
        
        for instruction in self.instructions:
            for account in instruction.accounts:
                if account.is_signer:
                    signer_accounts.add(account.pubkey)
                if account.is_writable:
                    writable_accounts.add(account.pubkey)
        
        # Order accounts according to Solana's specification:
        # 1. Required signers (writable signers)
        # 2. Readonly signers
        # 3. Writable non-signers
        # 4. Readonly non-signers
        
        account_keys = []
        
        # Required signers (writable signers)
        required_signers = sorted(signer_accounts & writable_accounts)
        account_keys.extend(required_signers)
        
        # Readonly signers
        readonly_signers = sorted(signer_accounts - writable_accounts)
        account_keys.extend(readonly_signers)
        
        # Writable non-signers (including programs if they're writable)
        writable_non_signers = sorted(writable_accounts - signer_accounts)
        account_keys.extend(writable_non_signers)
        
        # Readonly non-signers
        readonly_non_signers = sorted(all_accounts - set(account_keys))
        account_keys.extend(readonly_non_signers)
        
        # Create account index mapping
        account_index = {key: i for i, key in enumerate(account_keys)}
        
        # Compile instructions
        compiled_instructions = []
        for instruction in self.instructions:
            try:
                compiled_instruction = CompiledInstruction(
                    program_id_index=account_index[instruction.program_id],
                    accounts=[account_index[acc.pubkey] for acc in instruction.accounts],
                    data=instruction.data
                )
                compiled_instructions.append(compiled_instruction)
            except KeyError as e:
                raise ValueError(f"Account not found in transaction: {e}")
        
        # Create message header
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


def sign_transaction(message: TransactionMessage, signers: List[SigningKey]) -> SolanaTransaction:
    """
    Sign a transaction message with the provided private keys.
    
    Args:
        message: Transaction message to sign
        signers: Private keys in the same order as required signers
        
    Returns:
        Fully signed transaction ready for submission
    """
    message_data = message.serialize()
    signatures = []
    
    for signer in signers:
        try:
            signature = signer.sign(message_data)
            signatures.append(signature.hex())
        except Exception as e:
            raise ValueError(f"Failed to sign transaction: {e}")
    
    return SolanaTransaction(signatures=signatures, message=message)


def generate_keypair() -> tuple[SigningKey, str]:
    """
    Generate a new Ed25519 keypair.
    
    Returns:
        Tuple of (private_key, public_key_hex)
    """
    private_key = SigningKey.generate(curve=SECP256k1)
    public_key_hex = private_key.verifying_key.to_string().hex()
    return private_key, public_key_hex


class TransactionSimulator:
    """
    Utility for simulating transaction execution without state changes.
    
    Useful for testing and fee estimation.
    """
    
    def __init__(self, account_registry):
        self.account_registry = account_registry
    
    def simulate_transaction(self, tx: SolanaTransaction) -> dict:
        """
        Simulate transaction execution and return metrics.
        
        Returns:
            Dictionary with simulation results including:
            - estimated_fee
            - accounts_accessed
            - compute_units_used
            - success (would it succeed)
        """
        result = {
            "tx_hash": tx.hash(),
            "estimated_fee": tx.calculate_fee(),
            "accounts_accessed": len(tx.message.account_keys),
            "instructions_count": len(tx.message.instructions),
            "signature_count": len(tx.signatures),
            "writable_accounts": list(tx.get_writable_accounts()),
            "readonly_accounts": list(tx.get_readonly_accounts()),
            "success": True,
            "errors": []
        }
        
        # Check if all accounts exist
        for account_key in tx.message.account_keys:
            if not self.account_registry.account_exists(account_key):
                result["success"] = False
                result["errors"].append(f"Account {account_key[:8]}... does not exist")
        
        # Check if fee payer has sufficient balance
        fee_payer = tx.get_fee_payer()
        fee_payer_balance = self.account_registry.get_account_balance(fee_payer)
        estimated_fee = result["estimated_fee"]
        
        if fee_payer_balance < estimated_fee:
            result["success"] = False
            result["errors"].append(f"Insufficient balance for fees: {fee_payer_balance} < {estimated_fee}")
        
        # Verify signatures
        if not tx.verify_signatures():
            result["success"] = False
            result["errors"].append("Invalid signatures")
        
        return result


# Utility functions for common transaction patterns

def create_transfer_instruction(from_pubkey: str, to_pubkey: str, 
                              lamports: int) -> Instruction:
    """Create a simple SOL transfer instruction."""
    data = bytearray()
    data.extend([2])  # Transfer instruction type
    data.extend(lamports.to_bytes(8, 'little'))
    
    return Instruction(
        program_id="11111111111111111111111111111111",  # System Program
        accounts=[
            AccountMeta(from_pubkey, is_signer=True, is_writable=True),
            AccountMeta(to_pubkey, is_signer=False, is_writable=True)
        ],
        data=bytes(data)
    )


def create_account_creation_instruction(payer: str, new_account: str,
                                      lamports: int, space: int, 
                                      owner: str) -> Instruction:
    """Create an account creation instruction."""
    data = bytearray()
    data.extend([0])  # CreateAccount instruction type
    data.extend(lamports.to_bytes(8, 'little'))
    data.extend(space.to_bytes(8, 'little'))
    data.extend(bytes.fromhex(owner))
    
    return Instruction(
        program_id="11111111111111111111111111111111",  # System Program
        accounts=[
            AccountMeta(payer, is_signer=True, is_writable=True),
            AccountMeta(new_account, is_signer=True, is_writable=True)
        ],
        data=bytes(data)
    ) 