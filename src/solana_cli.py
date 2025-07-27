#!/usr/bin/env python3
"""
Solana Blockchain CLI

A command-line interface for interacting with our Solana-like blockchain implementation.
This provides all the essential functionality for running a Solana node and interacting
with the blockchain, demonstrating the complete system in action.

Usage:
    python solana_cli.py start-validator    # Start a validator node
    python solana_cli.py transfer <from> <to> <amount>    # Send SOL
    python solana_cli.py balance <account>   # Check balance
    python solana_cli.py status             # Show blockchain status
    python solana_cli.py demo               # Run interactive demo

Author: Elegant implementation in Peter Norvig's style
"""

import argparse
import asyncio
import json
import sys
import time
import os
import signal
from typing import Dict, Any, Optional
from pathlib import Path

# Core imports
from core.blockchain import SolanaBlockchain
from core.transactions import generate_keypair, create_transfer_instruction, TransactionBuilder, sign_transaction
from core.poh import benchmark_poh


class ValidatorStateManager:
    """Manages validator state across different processes/terminals."""
    
    def __init__(self, state_dir: Path = None):
        self.state_dir = state_dir or Path.home() / ".solana-validator"
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "validator.json"
        self.pid_file = self.state_dir / "validator.pid"
    
    def write_state(self, node_id: str, pid: int, port: int = 8899):
        """Write validator state to file."""
        try:
            state = {
                'node_id': node_id,
                'pid': pid,
                'port': port,
                'start_time': time.time(),
                'status': 'running'
            }
            
            print(f"📁 Writing validator state to {self.state_file}")
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            with open(self.pid_file, 'w') as f:
                f.write(str(pid))
                
            print(f"✅ Validator state written successfully (PID: {pid})")
        except Exception as e:
            print(f"❌ Failed to write validator state: {e}")
            
    def read_state(self) -> Optional[Dict[str, Any]]:
        """Read validator state from file."""
        if not self.state_file.exists():
            return None
            
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def is_validator_running(self) -> bool:
        """Check if validator is currently running."""
        state = self.read_state()
        if not state:
            print(f"🔍 No validator state found at {self.state_file}")
            return False
            
        pid = state.get('pid')
        if not pid:
            print("⚠️ No PID found in validator state")
            return False
            
        # Check if process is still running
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            print(f"✅ Found running validator (PID: {pid})")
            return True
        except (OSError, ProcessLookupError):
            # Process doesn't exist, clean up stale files
            print(f"🧹 Process {pid} not running, cleaning up stale files")
            self.cleanup_state()
            return False
    
    def cleanup_state(self):
        """Clean up state files."""
        for file_path in [self.state_file, self.pid_file]:
            if file_path.exists():
                file_path.unlink()
    
    def get_validator_info(self) -> Optional[Dict[str, Any]]:
        """Get running validator information."""
        if self.is_validator_running():
            return self.read_state()
        return None


class SolanaNode:
    """
    A complete Solana validator node implementation.
    
    This orchestrates all the components to create a functioning
    blockchain node that can produce blocks, process transactions,
    and participate in consensus.
    """
    
    def __init__(self, node_id: str = "validator-1"):
        """Initialize the Solana node."""
        print(f"Initializing Solana node: {node_id}")
        
        self.node_id = node_id
        self.blockchain = SolanaBlockchain()
        self.is_running = False
        self.keypairs = {}  # Store generated keypairs
        self.state_manager = ValidatorStateManager()
        
        # Node configuration
        self.block_production_interval = 0.4  # 400ms slots
        self.auto_produce_blocks = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"Node {node_id} initialized successfully")
        print(f"Genesis leader: {self.blockchain.genesis_leader[:16]}...")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
        
    def stop(self):
        """Stop the validator and clean up."""
        self.is_running = False
        self.state_manager.cleanup_state()
        print("✅ Validator stopped")
    
    async def start(self):
        """Start the validator node."""
        print(f"Starting validator node {self.node_id}...")
        self.is_running = True
        
        # Write state file so other terminals can detect this validator
        self.state_manager.write_state(self.node_id, os.getpid())
        
        # Start background tasks
        tasks = [
            self.block_production_loop(),
            self.mempool_monitoring_loop(),
            self.performance_monitoring_loop()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print(f"\n🛑 Shutting down node {self.node_id}...")
            self.stop()
    
    async def block_production_loop(self):
        """Main block production loop."""
        print("Starting block production...")
        
        while self.is_running:
            try:
                if self.auto_produce_blocks:
                    # Check if there are transactions to process
                    if (self.blockchain.mempool.get_pending_count() > 0 or 
                        time.time() % 2 < 0.4):  # Produce empty blocks occasionally
                        
                        block = self.blockchain.produce_block()
                        if block:
                            print(f"✓ Produced block {block.slot}")
                
                await asyncio.sleep(self.block_production_interval)
                
            except Exception as e:
                print(f"Error in block production: {e}")
                await asyncio.sleep(1.0)
    
    async def mempool_monitoring_loop(self):
        """Monitor mempool and clean up invalid transactions."""
        while self.is_running:
            try:
                # Clean up invalid transactions every 10 seconds
                cleaned = self.blockchain.mempool.cleanup_invalid_transactions()
                if cleaned > 0:
                    print(f"Cleaned {cleaned} invalid transactions from mempool")
                
                await asyncio.sleep(10.0)
                
            except Exception as e:
                print(f"Error in mempool monitoring: {e}")
                await asyncio.sleep(1.0)
    
    async def performance_monitoring_loop(self):
        """Monitor and report performance metrics."""
        while self.is_running:
            try:
                # Report status every 30 seconds
                await asyncio.sleep(30.0)
                if self.is_running:
                    self.print_performance_summary()
                
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                await asyncio.sleep(1.0)
    
    def print_performance_summary(self):
        """Print a concise performance summary."""
        stats = self.blockchain.get_blockchain_stats()
        poh_info = self.blockchain.poh.get_rate_info()
        
        print(f"\n📊 Performance Summary (Node: {self.node_id})")
        print(f"   Slot: {stats.current_slot} | TPS: {stats.average_tps:.1f} | "
              f"Blocks: {stats.total_blocks} | Pending: {self.blockchain.mempool.get_pending_count()}")
        print(f"   PoH: {poh_info['entries_per_second']:.1f} entries/sec | "
              f"Success rate: {(stats.successful_transactions/stats.total_transactions*100):.1f}%" if stats.total_transactions > 0 else "N/A")


class SolanaCLI:
    """
    Command-line interface for Solana blockchain operations.
    
    Provides a clean, intuitive interface for all blockchain interactions
    with helpful error messages and progress indicators.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.node: Optional[SolanaNode] = None
        self.wallet_path = Path.home() / ".solana-wallet.json"
        self.keypairs = self.load_wallet()
        self.state_manager = ValidatorStateManager()
    
    def load_wallet(self) -> Dict[str, Any]:
        """Load stored keypairs from wallet file."""
        if self.wallet_path.exists():
            try:
                with open(self.wallet_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load wallet: {e}")
        return {}
    
    def save_wallet(self):
        """Save keypairs to wallet file."""
        try:
            self.wallet_path.parent.mkdir(exist_ok=True)
            with open(self.wallet_path, 'w') as f:
                json.dump(self.keypairs, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save wallet: {e}")
    
    def get_or_create_keypair(self, name: str) -> tuple:
        """Get existing keypair or create a new one."""
        if name in self.keypairs:
            from ecdsa import SigningKey, SECP256k1
            private_key = SigningKey.from_string(bytes.fromhex(self.keypairs[name]['private_key']), curve=SECP256k1)
            public_key = self.keypairs[name]['public_key']
            return private_key, public_key
        else:
            # Create new keypair
            private_key, public_key = generate_keypair()
            self.keypairs[name] = {
                'private_key': private_key.to_string().hex(),
                'public_key': public_key
            }
            self.save_wallet()
            print(f"Created new keypair for {name}: {public_key[:16]}...")
            return private_key, public_key
    
    def get_running_validator(self) -> Optional[SolanaNode]:
        """Check if there's a running validator and connect to it."""
        if self.node and self.node.is_running:
            return self.node
        
        # Check if there's a validator running in another process
        if self.state_manager.is_validator_running():
            validator_info = self.state_manager.get_validator_info()
            if validator_info:
                print(f"📡 Detected running validator: {validator_info['node_id']} (PID: {validator_info['pid']})")
                print("💡 Note: In this demo, CLI operations create a separate blockchain instance.")
                print("    In a production implementation, this would connect via RPC to the running validator.")
                
                # For this educational demo, create a fresh blockchain instance
                # This allows CLI operations to work, though it's not the same state as the running validator
                self.node = SolanaNode(validator_info['node_id'])
                return self.node
        
        return None

    async def start_validator(self, node_id: str = "validator-1"):
        """Start a validator node."""
        # Check if a validator is already running
        if self.state_manager.is_validator_running():
            validator_info = self.state_manager.get_validator_info()
            print(f"❌ Validator '{validator_info['node_id']}' is already running (PID: {validator_info['pid']})")
            print("   Stop the existing validator first or use a different node ID.")
            return
        
        print("🚀 Starting Solana Validator Node")
        print("=" * 50)
        
        self.node = SolanaNode(node_id)
        
        # Create some initial accounts for testing
        print("\n📝 Setting up test accounts...")
        alice_keypair = self.get_or_create_keypair("alice")
        bob_keypair = self.get_or_create_keypair("bob")
        
        # Fund Alice's account
        alice_pubkey = alice_keypair[1]
        if not self.node.blockchain.accounts.account_exists(alice_pubkey):
            self.node.blockchain.create_account(alice_pubkey, lamports=10_000_000_000)  # 10 SOL
            print(f"💰 Funded Alice with 10 SOL: {alice_pubkey[:16]}...")
        
        # Create Bob's account
        bob_pubkey = bob_keypair[1]
        if not self.node.blockchain.accounts.account_exists(bob_pubkey):
            self.node.blockchain.create_account(bob_pubkey)
            print(f"🆕 Created Bob's account: {bob_pubkey[:16]}...")
        
        print(f"\n✅ Validator ready! Starting block production...")
        print(f"💡 Tip: Open another terminal and run 'python solana_cli.py demo' to see it in action!")
        
        await self.node.start()
    
    def transfer(self, from_name: str, to_name: str, amount: float):
        """Transfer SOL between accounts."""
        validator = self.get_running_validator()
        if not validator:
            print("❌ No running node. Start a validator first.")
            return
        
        print(f"💸 Transferring {amount} SOL from {from_name} to {to_name}")
        
        # Get keypairs
        from_keypair = self.get_or_create_keypair(from_name)
        to_keypair = self.get_or_create_keypair(to_name)
        
        # Convert SOL to lamports
        lamports = int(amount * 1_000_000_000)
        
        # Create and submit transaction
        transaction = validator.blockchain.create_transfer_transaction(
            from_keypair, to_keypair[1], lamports
        )
        
        if transaction:
            success = validator.blockchain.add_transaction(transaction)
            if success:
                print(f"✅ Transaction submitted: {transaction.hash()[:16]}...")
                print(f"⏳ Waiting for confirmation...")
                
                # Wait for confirmation
                self.wait_for_confirmation(transaction.hash(), validator)
            else:
                print(f"❌ Transaction rejected")
        else:
            print(f"❌ Failed to create transaction")
    
    def balance(self, account_name: str):
        """Check account balance."""
        validator = self.get_running_validator()
        if not validator:
            print("❌ No running node. Start a validator first.")
            return
        
        keypair = self.get_or_create_keypair(account_name)
        pubkey = keypair[1]
        
        balance_lamports = validator.blockchain.get_balance(pubkey)
        balance_sol = balance_lamports / 1_000_000_000
        
        print(f"💰 Balance for {account_name}:")
        print(f"   Address: {pubkey}")
        print(f"   Balance: {balance_sol:.6f} SOL ({balance_lamports:,} lamports)")
    
    def status(self):
        """Show blockchain status."""
        validator = self.get_running_validator()
        if not validator:
            print("❌ No running node. Start a validator first.")
            return
        
        validator.blockchain.display_status()
    
    def wait_for_confirmation(self, tx_hash: str, validator: SolanaNode, timeout: int = 30):
        """Wait for transaction confirmation."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = validator.blockchain.get_transaction_status(tx_hash)
            
            if status['status'] == 'confirmed':
                print(f"✅ Transaction confirmed in slot {status['slot']}")
                return True
            elif status['status'] == 'failed':
                print(f"❌ Transaction failed: {status['error']}")
                return False
            
            time.sleep(0.5)
        
        print(f"⏰ Transaction confirmation timeout")
        return False
    
    def demo(self):
        """Run an interactive demo."""
        validator = self.get_running_validator()
        if not validator:
            print("❌ No running node. Start a validator first with:")
            print("   python solana_cli.py start-validator")
            return
        
        print("🎮 Interactive Solana Demo")
        print("=" * 40)
        
        while True:
            print(f"\nCurrent slot: {validator.blockchain.current_slot()}")
            print(f"Pending transactions: {validator.blockchain.mempool.get_pending_count()}")
            
            print("\nChoose an action:")
            print("1. Send transaction")
            print("2. Check balance")
            print("3. Show status")
            print("4. Generate transactions")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                from_name = input("From account: ").strip() or "alice"
                to_name = input("To account: ").strip() or "bob"
                amount = float(input("Amount (SOL): ").strip() or "1.0")
                self.transfer(from_name, to_name, amount)
            
            elif choice == '2':
                account = input("Account name: ").strip() or "alice"
                self.balance(account)
            
            elif choice == '3':
                self.status()
            
            elif choice == '4':
                count = int(input("Number of transactions to generate: ").strip() or "5")
                self.generate_test_transactions(count, validator)
            
            elif choice == '5':
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    def generate_test_transactions(self, count: int, validator: SolanaNode):
        """Generate test transactions for demonstration."""
        print(f"🔄 Generating {count} test transactions...")
        
        accounts = ['alice', 'bob', 'charlie', 'dave', 'eve']
        
        for i in range(count):
            # Ensure all accounts exist with some balance
            for name in accounts:
                keypair = self.get_or_create_keypair(name)
                pubkey = keypair[1]
                if not validator.blockchain.accounts.account_exists(pubkey):
                    validator.blockchain.create_account(pubkey, lamports=1_000_000_000)  # 1 SOL
            
            # Create random transfer
            import random
            from_name = random.choice(accounts)
            to_name = random.choice([acc for acc in accounts if acc != from_name])
            amount = random.uniform(0.01, 0.1)  # 0.01 to 0.1 SOL
            
            print(f"  {i+1}/{count}: {from_name} → {to_name} ({amount:.3f} SOL)")
            
            from_keypair = self.get_or_create_keypair(from_name)
            to_keypair = self.get_or_create_keypair(to_name)
            
            transaction = validator.blockchain.create_transfer_transaction(
                from_keypair, to_keypair[1], int(amount * 1_000_000_000)
            )
            
            if transaction:
                validator.blockchain.add_transaction(transaction)
            
            time.sleep(0.1)  # Small delay between transactions
        
        print(f"✅ Generated {count} transactions")
    
    def benchmark(self):
        """Run performance benchmarks."""
        print("⚡ Running Solana Performance Benchmarks")
        print("=" * 50)
        
        # PoH benchmark
        print("\n📊 Proof of History Benchmark:")
        poh_results = benchmark_poh(duration=5.0, iterations=1000)
        print(f"   Entries per second: {poh_results['entries_per_second']:.2f}")
        print(f"   Hash rate: {poh_results['iterations_per_second']:.0f} hashes/sec")
        
        validator = self.get_running_validator()
        if validator:
            # Transaction throughput test
            print(f"\n🚀 Transaction Throughput Test:")
            start_time = time.time()
            self.generate_test_transactions(100, validator)
            
            # Wait for processing
            time.sleep(5.0)
            
            stats = validator.blockchain.get_blockchain_stats()
            elapsed = time.time() - start_time
            
            print(f"   Processed: {stats.total_transactions} transactions")
            print(f"   Time: {elapsed:.2f} seconds")
            print(f"   Average TPS: {stats.total_transactions / elapsed:.2f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Solana Blockchain CLI - Educational Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python solana_cli.py start-validator           # Start validator node
  python solana_cli.py transfer alice bob 1.5    # Send 1.5 SOL
  python solana_cli.py balance alice              # Check balance
  python solana_cli.py status                     # Show blockchain status
  python solana_cli.py demo                       # Interactive demo
  python solana_cli.py benchmark                  # Performance tests
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start validator command
    start_parser = subparsers.add_parser('start-validator', help='Start a validator node')
    start_parser.add_argument('--node-id', default='validator-1', help='Node identifier')
    
    # Transfer command
    transfer_parser = subparsers.add_parser('transfer', help='Transfer SOL between accounts')
    transfer_parser.add_argument('from_account', help='Source account name')
    transfer_parser.add_argument('to_account', help='Destination account name')
    transfer_parser.add_argument('amount', type=float, help='Amount in SOL')
    
    # Balance command
    balance_parser = subparsers.add_parser('balance', help='Check account balance')
    balance_parser.add_argument('account', help='Account name')
    
    # Status command
    subparsers.add_parser('status', help='Show blockchain status')
    
    # Demo command
    subparsers.add_parser('demo', help='Run interactive demo')
    
    # Benchmark command
    subparsers.add_parser('benchmark', help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    cli = SolanaCLI()
    
    try:
        if args.command == 'start-validator':
            asyncio.run(cli.start_validator(args.node_id))
        
        elif args.command == 'transfer':
            cli.transfer(args.from_account, args.to_account, args.amount)
        
        elif args.command == 'balance':
            cli.balance(args.account)
        
        elif args.command == 'status':
            cli.status()
        
        elif args.command == 'demo':
            cli.demo()
        
        elif args.command == 'benchmark':
            cli.benchmark()
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 