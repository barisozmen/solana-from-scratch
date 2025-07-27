# Stage 6: Networking and Gossip Protocol

> *"The network is the computer."* - John Gage, Sun Microsystems

## Objectives

This stage transforms our single-process blockchain into a truly distributed system with peer-to-peer networking:

- Implement P2P networking using Python's `asyncio` for non-blocking I/O
- Create gossip protocol for validator discovery and information sharing
- Add Turbine-inspired block propagation with efficient fan-out
- Implement Gulf Stream-like transaction forwarding to predicted leaders
- Run multiple nodes that achieve consensus across the network

## Why Distributed Networking?

### Centralization Problems
- **Single point of failure**: One node controls everything
- **Censorship risk**: Central authority can block transactions
- **Trust requirements**: Must trust the single operator
- **Scalability limits**: Bounded by single machine capacity

### P2P Network Benefits
- **Decentralization**: No single point of control
- **Fault tolerance**: Network survives node failures
- **Censorship resistance**: Hard to block distributed network
- **Global accessibility**: Anyone can participate

## Key Concepts

### 1. Gossip Protocol
Information spreads through the network like rumors:
- **Random peer selection**: Each node talks to a few random neighbors
- **Exponential propagation**: Information reaches all nodes quickly
- **Fault tolerance**: Works even with node failures
- **Low overhead**: Efficient bandwidth usage

### 2. Turbine Block Propagation
Inspired by Solana's Turbine protocol:
- **Block sharding**: Split blocks into chunks
- **Tree topology**: Organized fan-out pattern
- **Parallel transmission**: Multiple chunks sent simultaneously
- **Reduced latency**: Faster than sequential broadcast

### 3. Gulf Stream Transaction Forwarding
Efficient transaction routing:
- **Leader prediction**: Use PoH to predict future leaders
- **Direct forwarding**: Send transactions to upcoming leaders
- **Reduced latency**: Skip intermediate hops
- **Mempool optimization**: Leaders get transactions faster

## Implementation

### Network Node Structure

```python
class NetworkNode:
    def __init__(self, node_id: str, port: int, validator: Validator = None):
        self.node_id = node_id
        self.port = port
        self.validator = validator
        self.blockchain = TowerBFTBlockchain()
        self.peers: Set[tuple[str, int]] = set()
        self.running = False
        
        # Network state
        self.known_validators: Dict[str, Validator] = {}
        self.pending_messages: List[Dict] = []
        
        # Add own validator if provided
        if validator:
            self.blockchain.stake_registry.add_validator(validator)
```

### Message Types

```python
@dataclass
class NetworkMessage:
    type: str           # Message type identifier
    sender_id: str      # Who sent this message
    data: Dict         # Message payload
    timestamp: float   # When message was created
    
    def to_json(self) -> str:
        return json.dumps({
            'type': self.type,
            'sender_id': self.sender_id,
            'data': self.data,
            'timestamp': self.timestamp
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'NetworkMessage':
        data = json.loads(json_str)
        return cls(**data)

# Message type constants
class MessageType:
    GOSSIP_VALIDATORS = "gossip_validators"
    TRANSACTION = "transaction"
    BLOCK_PROPOSAL = "block_proposal"
    VOTE = "vote"
    BLOCK_CHUNK = "block_chunk"
    PEER_DISCOVERY = "peer_discovery"
```

### Gossip Implementation

```python
class GossipProtocol:
    def __init__(self, node: 'NetworkNode'):
        self.node = node
        self.gossip_interval = 2.0  # seconds
        self.fanout = 3  # number of peers to gossip to
    
    async def start_gossiping(self):
        """Start periodic gossip process."""
        while self.node.running:
            await self.gossip_round()
            await asyncio.sleep(self.gossip_interval)
    
    async def gossip_round(self):
        """Perform one round of gossip with random peers."""
        if not self.node.peers:
            return
        
        # Select random peers for this round
        selected_peers = random.sample(
            list(self.node.peers), 
            min(self.fanout, len(self.node.peers))
        )
        
        # Gossip validator information
        await self._gossip_validators(selected_peers)
        
        # Gossip recent blocks
        await self._gossip_recent_blocks(selected_peers)
    
    async def _gossip_validators(self, peers: List[tuple[str, int]]):
        """Share validator registry with peers."""
        validator_data = {}
        for vid, validator in self.node.blockchain.stake_registry.validators.items():
            validator_data[vid] = {
                'id': validator.id,
                'stake': validator.stake,
                'public_key': validator.public_key
                # Note: private_key is never shared
            }
        
        message = NetworkMessage(
            type=MessageType.GOSSIP_VALIDATORS,
            sender_id=self.node.node_id,
            data={'validators': validator_data},
            timestamp=time.time()
        )
        
        await self._send_to_peers(message, peers)
    
    async def _gossip_recent_blocks(self, peers: List[tuple[str, int]]):
        """Share recent blocks with peers."""
        canonical_chain = self.node.blockchain.fork_manager.get_canonical_chain()
        recent_blocks = canonical_chain[-5:]  # Last 5 blocks
        
        for block in recent_blocks:
            await self._propagate_block_turbine_style(block, peers)
    
    async def _send_to_peers(self, message: NetworkMessage, peers: List[tuple[str, int]]):
        """Send message to specified peers."""
        tasks = []
        for host, port in peers:
            task = self._send_message(host, port, message)
            tasks.append(task)
        
        # Send to all peers concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
```

### Turbine Block Propagation

```python
class TurbineProtocol:
    def __init__(self, node: 'NetworkNode'):
        self.node = node
        self.chunk_size = 1024  # bytes per chunk
        self.fanout = 2  # children per node in tree
    
    async def propagate_block(self, block: Block):
        """Propagate block using Turbine-style sharding."""
        # Serialize block
        block_data = self._serialize_block(block)
        
        # Split into chunks
        chunks = self._create_chunks(block_data, block.hash())
        
        # Propagate each chunk
        await self._propagate_chunks(chunks)
    
    def _create_chunks(self, data: bytes, block_hash: str) -> List[Dict]:
        """Split block data into chunks."""
        chunks = []
        chunk_size = self.chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk_data = data[i:i + chunk_size]
            chunk = {
                'block_hash': block_hash,
                'chunk_index': i // chunk_size,
                'total_chunks': (len(data) + chunk_size - 1) // chunk_size,
                'data': chunk_data.hex()
            }
            chunks.append(chunk)
        
        return chunks
    
    async def _propagate_chunks(self, chunks: List[Dict]):
        """Send chunks to different peers in parallel."""
        if not self.node.peers:
            return
        
        peers = list(self.node.peers)
        
        # Send each chunk to different subset of peers
        for i, chunk in enumerate(chunks):
            # Select peers for this chunk (round-robin)
            selected_peers = []
            for j in range(self.fanout):
                peer_index = (i * self.fanout + j) % len(peers)
                selected_peers.append(peers[peer_index])
            
            message = NetworkMessage(
                type=MessageType.BLOCK_CHUNK,
                sender_id=self.node.node_id,
                data=chunk,
                timestamp=time.time()
            )
            
            await self._send_to_peers(message, selected_peers)
    
    def _serialize_block(self, block: Block) -> bytes:
        """Serialize block for transmission."""
        block_dict = {
            'index': block.index,
            'poh_entry': {'counter': block.poh_entry.counter, 'hash': block.poh_entry.hash},
            'transactions': [
                {
                    'sender': tx.sender,
                    'receiver': tx.receiver,
                    'amount': tx.amount,
                    'signature': tx.signature
                } for tx in block.transactions
            ],
            'previous_hash': block.previous_hash,
            'leader_id': block.leader_id,
            'votes': [
                {
                    'validator_id': vote.validator_id,
                    'block_hash': vote.block_hash,
                    'slot': vote.slot,
                    'signature': vote.signature
                } for vote in block.votes
            ]
        }
        return json.dumps(block_dict).encode()
```

### Gulf Stream Transaction Forwarding

```python
class GulfStreamProtocol:
    def __init__(self, node: 'NetworkNode'):
        self.node = node
        self.leader_cache: Dict[int, str] = {}  # slot -> leader_id
    
    async def forward_transaction(self, tx: Transaction):
        """Forward transaction to predicted upcoming leaders."""
        # Predict next few leaders
        upcoming_leaders = self._predict_upcoming_leaders(3)
        
        # Find peers for these leaders
        target_peers = self._find_leader_peers(upcoming_leaders)
        
        # Forward transaction
        message = NetworkMessage(
            type=MessageType.TRANSACTION,
            sender_id=self.node.node_id,
            data={
                'sender': tx.sender,
                'receiver': tx.receiver,
                'amount': tx.amount,
                'signature': tx.signature
            },
            timestamp=time.time()
        )
        
        await self._send_to_peers(message, target_peers)
    
    def _predict_upcoming_leaders(self, num_slots: int) -> List[str]:
        """Predict leaders for upcoming slots."""
        leaders = []
        current_poh_hash = self.node.blockchain.poh.current_hash
        
        # Simulate future PoH progression
        for i in range(1, num_slots + 1):
            # Predict future PoH hash (simplified)
            future_hash = current_poh_hash
            for _ in range(i * 1000):  # Simulate iterations
                future_hash = hashlib.sha256(future_hash.encode()).hexdigest()
            
            # Select leader for that hash
            try:
                leader_id = self.node.blockchain.stake_registry.select_leader(future_hash)
                leaders.append(leader_id)
            except ValueError:
                pass  # No validators
        
        return leaders
    
    def _find_leader_peers(self, leader_ids: List[str]) -> List[tuple[str, int]]:
        """Find network peers that correspond to predicted leaders."""
        # Simplified: return random peers since we don't have leader->peer mapping
        # In real implementation, maintain validator->peer mapping
        available_peers = list(self.node.peers)
        if not available_peers:
            return []
        
        # Return subset of peers (assuming some might be the leaders)
        return random.sample(available_peers, min(len(leader_ids), len(available_peers)))
```

## Complete Implementation

Create `stage6_network_blockchain.py`:

```python
import asyncio
import json
import hashlib
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple, Optional
from ecdsa import SigningKey, VerifyingKey, SECP256k1

# Import previous stage components (simplified for demo)
from stage5_tower_bft import TowerBFTBlockchain, Validator, Transaction, Block, Vote

@dataclass
class NetworkMessage:
    type: str
    sender_id: str
    data: Dict
    timestamp: float
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'NetworkMessage':
        data = json.loads(json_str)
        return cls(**data)

class MessageType:
    GOSSIP_VALIDATORS = "gossip_validators"
    TRANSACTION = "transaction"
    BLOCK_PROPOSAL = "block_proposal"
    VOTE = "vote"
    BLOCK_CHUNK = "block_chunk"
    PEER_DISCOVERY = "peer_discovery"

class NetworkNode:
    def __init__(self, node_id: str, port: int, validator: Validator = None):
        self.node_id = node_id
        self.port = port
        self.validator = validator
        self.blockchain = TowerBFTBlockchain()
        self.peers: Set[Tuple[str, int]] = set()
        self.running = False
        self.server = None
        
        # Add own validator
        if validator:
            self.blockchain.stake_registry.add_validator(validator)
            self.blockchain.set_balance(validator.public_key, 100.0)
        
        # Network protocols
        self.gossip_protocol = GossipProtocol(self)
        self.turbine_protocol = TurbineProtocol(self)
        self.gulf_stream_protocol = GulfStreamProtocol(self)
        
        # Block reconstruction
        self.partial_blocks: Dict[str, Dict] = {}  # block_hash -> chunks

    async def start(self):
        """Start the network node."""
        self.running = True
        
        # Start server
        self.server = await asyncio.start_server(
            self.handle_connection, 
            'localhost', 
            self.port
        )
        
        print(f"Node {self.node_id[:8]}... started on port {self.port}")
        
        # Start gossip protocol
        asyncio.create_task(self.gossip_protocol.start_gossiping())
        
        # Start slot progression (if we're a validator)
        if self.validator:
            asyncio.create_task(self._slot_progression())
        
        # Serve forever
        async with self.server:
            await self.server.serve_forever()

    async def stop(self):
        """Stop the network node."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def handle_connection(self, reader, writer):
        """Handle incoming network connections."""
        try:
            data = await reader.read(8192)
            if data:
                message = NetworkMessage.from_json(data.decode())
                await self._process_message(message)
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_message(self, message: NetworkMessage):
        """Process received network message."""
        if message.type == MessageType.GOSSIP_VALIDATORS:
            await self._handle_gossip_validators(message)
        elif message.type == MessageType.TRANSACTION:
            await self._handle_transaction(message)
        elif message.type == MessageType.BLOCK_PROPOSAL:
            await self._handle_block_proposal(message)
        elif message.type == MessageType.VOTE:
            await self._handle_vote(message)
        elif message.type == MessageType.BLOCK_CHUNK:
            await self._handle_block_chunk(message)
        elif message.type == MessageType.PEER_DISCOVERY:
            await self._handle_peer_discovery(message)

    async def _handle_gossip_validators(self, message: NetworkMessage):
        """Handle validator gossip message."""
        validators_data = message.data.get('validators', {})
        
        for vid, val_data in validators_data.items():
            if vid not in self.blockchain.stake_registry.validators:
                # Create validator without private key (we don't have it)
                validator = Validator(
                    id=val_data['id'],
                    stake=val_data['stake'],
                    public_key=val_data['public_key'],
                    private_key=""  # Empty for remote validators
                )
                self.blockchain.stake_registry.add_validator(validator)
                print(f"Learned about validator {vid[:8]}... (stake: {validator.stake})")

    async def _handle_transaction(self, message: NetworkMessage):
        """Handle incoming transaction."""
        tx_data = message.data
        tx = Transaction(
            sender=tx_data['sender'],
            receiver=tx_data['receiver'],
            amount=tx_data['amount'],
            signature=tx_data['signature']
        )
        
        # Add to our mempool
        if self.blockchain.add_transaction(tx):
            print(f"Received transaction: {tx.amount} from {tx.sender[:8]}...")

    async def _handle_block_proposal(self, message: NetworkMessage):
        """Handle block proposal."""
        # In full implementation, validate and vote on block
        print(f"Received block proposal from {message.sender_id[:8]}...")

    async def _handle_vote(self, message: NetworkMessage):
        """Handle vote message."""
        # In full implementation, collect votes for consensus
        print(f"Received vote from {message.sender_id[:8]}...")

    async def _handle_block_chunk(self, message: NetworkMessage):
        """Handle block chunk for reconstruction."""
        chunk_data = message.data
        block_hash = chunk_data['block_hash']
        
        # Initialize block reconstruction
        if block_hash not in self.partial_blocks:
            self.partial_blocks[block_hash] = {
                'chunks': {},
                'total_chunks': chunk_data['total_chunks']
            }
        
        # Store chunk
        chunk_index = chunk_data['chunk_index']
        self.partial_blocks[block_hash]['chunks'][chunk_index] = chunk_data['data']
        
        # Check if we have all chunks
        partial_block = self.partial_blocks[block_hash]
        if len(partial_block['chunks']) == partial_block['total_chunks']:
            await self._reconstruct_block(block_hash)

    async def _reconstruct_block(self, block_hash: str):
        """Reconstruct block from chunks."""
        partial_block = self.partial_blocks[block_hash]
        
        # Combine chunks in order
        combined_data = ""
        for i in range(partial_block['total_chunks']):
            if i in partial_block['chunks']:
                combined_data += partial_block['chunks'][i]
        
        try:
            # Reconstruct block
            block_bytes = bytes.fromhex(combined_data)
            block_dict = json.loads(block_bytes.decode())
            
            # Convert back to Block object (simplified)
            print(f"Reconstructed block {block_hash[:8]}... from {partial_block['total_chunks']} chunks")
            
            # Clean up
            del self.partial_blocks[block_hash]
            
        except Exception as e:
            print(f"Failed to reconstruct block: {e}")

    async def _handle_peer_discovery(self, message: NetworkMessage):
        """Handle peer discovery message."""
        peers_data = message.data.get('peers', [])
        for host, port in peers_data:
            if (host, port) != ('localhost', self.port):  # Don't add ourselves
                self.peers.add((host, port))

    async def connect_to_peer(self, host: str, port: int):
        """Connect to a new peer."""
        if (host, port) not in self.peers and port != self.port:
            self.peers.add((host, port))
            
            # Send peer discovery message
            message = NetworkMessage(
                type=MessageType.PEER_DISCOVERY,
                sender_id=self.node_id,
                data={'peers': [('localhost', self.port)]},
                timestamp=time.time()
            )
            
            await self._send_message(host, port, message)
            print(f"Connected to peer {host}:{port}")

    async def _send_message(self, host: str, port: int, message: NetworkMessage):
        """Send message to specific peer."""
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(message.to_json().encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            # Remove failed peer
            self.peers.discard((host, port))

    async def _slot_progression(self):
        """Handle slot progression for validators."""
        while self.running:
            # Only advance if we have validators
            if self.blockchain.stake_registry.validators:
                leader_id = self.blockchain.advance_slot()
                
                # If we're the leader, propagate the new block
                if leader_id == self.validator.id:
                    canonical_chain = self.blockchain.fork_manager.get_canonical_chain()
                    latest_block = canonical_chain[-1]
                    await self.turbine_protocol.propagate_block(latest_block)
                    print(f"Slot {self.blockchain.current_slot}: We are the leader!")
                else:
                    print(f"Slot {self.blockchain.current_slot}: Leader is {leader_id[:8]}...")
            
            await asyncio.sleep(3.0)  # Slot duration

    def display_network_state(self):
        """Display current network state."""
        print(f"\n{'='*50}")
        print(f"NETWORK NODE STATE - {self.node_id[:8]}...")
        print(f"{'='*50}")
        print(f"Port: {self.port}")
        print(f"Connected peers: {len(self.peers)}")
        print(f"Known validators: {len(self.blockchain.stake_registry.validators)}")
        print(f"Current slot: {self.blockchain.current_slot}")
        
        canonical_chain = self.blockchain.fork_manager.get_canonical_chain()
        print(f"Chain length: {len(canonical_chain)}")
        print(f"Mempool size: {self.blockchain.mempool.size()}")

class GossipProtocol:
    def __init__(self, node: NetworkNode):
        self.node = node
        self.gossip_interval = 5.0
        self.fanout = 2

    async def start_gossiping(self):
        while self.node.running:
            await asyncio.sleep(self.gossip_interval)
            await self.gossip_round()

    async def gossip_round(self):
        if not self.node.peers:
            return

        selected_peers = random.sample(
            list(self.node.peers),
            min(self.fanout, len(self.node.peers))
        )

        await self._gossip_validators(selected_peers)

    async def _gossip_validators(self, peers: List[Tuple[str, int]]):
        validator_data = {}
        for vid, validator in self.node.blockchain.stake_registry.validators.items():
            validator_data[vid] = {
                'id': validator.id,
                'stake': validator.stake,
                'public_key': validator.public_key
            }

        message = NetworkMessage(
            type=MessageType.GOSSIP_VALIDATORS,
            sender_id=self.node.node_id,
            data={'validators': validator_data},
            timestamp=time.time()
        )

        for host, port in peers:
            await self.node._send_message(host, port, message)

class TurbineProtocol:
    def __init__(self, node: NetworkNode):
        self.node = node
        self.chunk_size = 512

    async def propagate_block(self, block: Block):
        if not self.node.peers:
            return

        # Serialize block (simplified)
        block_data = json.dumps({
            'index': block.index,
            'hash': block.hash(),
            'leader': block.leader_id,
            'transactions': len(block.transactions)
        }).encode()

        # Create chunks
        chunks = []
        for i in range(0, len(block_data), self.chunk_size):
            chunk_data = block_data[i:i + self.chunk_size]
            chunk = {
                'block_hash': block.hash(),
                'chunk_index': i // self.chunk_size,
                'total_chunks': (len(block_data) + self.chunk_size - 1) // self.chunk_size,
                'data': chunk_data.hex()
            }
            chunks.append(chunk)

        # Send chunks to peers
        peers = list(self.node.peers)
        for i, chunk in enumerate(chunks):
            peer_index = i % len(peers)
            host, port = peers[peer_index]
            
            message = NetworkMessage(
                type=MessageType.BLOCK_CHUNK,
                sender_id=self.node.node_id,
                data=chunk,
                timestamp=time.time()
            )
            
            await self.node._send_message(host, port, message)

class GulfStreamProtocol:
    def __init__(self, node: NetworkNode):
        self.node = node

    async def forward_transaction(self, tx: Transaction):
        # Simplified: forward to all peers
        if not self.node.peers:
            return

        message = NetworkMessage(
            type=MessageType.TRANSACTION,
            sender_id=self.node.node_id,
            data={
                'sender': tx.sender,
                'receiver': tx.receiver,
                'amount': tx.amount,
                'signature': tx.signature
            },
            timestamp=time.time()
        )

        # Forward to subset of peers
        selected_peers = random.sample(
            list(self.node.peers),
            min(2, len(self.node.peers))
        )

        for host, port in selected_peers:
            await self.node._send_message(host, port, message)

# Helper functions
def generate_validator(stake: float) -> Tuple[str, Validator]:
    sk = SigningKey.generate(curve=SECP256k1)
    pk = sk.verifying_key.to_string().hex()
    sk_hex = sk.to_string().hex()
    
    validator = Validator(
        id=pk,
        stake=stake,
        public_key=pk,
        private_key=sk_hex
    )
    
    return pk, validator

# Test network
async def test_network():
    """Test distributed network with multiple nodes."""
    print("Testing Distributed Blockchain Network")
    print("=" * 50)
    
    # Create validators
    validators = []
    for i, stake in enumerate([1000, 600, 400]):
        pk, validator = generate_validator(stake)
        validators.append(validator)
        print(f"Validator {i+1}: {pk[:8]}... (stake: {stake})")
    
    # Create nodes
    nodes = []
    ports = [8001, 8002, 8003]
    
    for i, (port, validator) in enumerate(zip(ports, validators)):
        node = NetworkNode(f"node_{i}", port, validator)
        nodes.append(node)
    
    # Start nodes
    node_tasks = []
    for node in nodes:
        task = asyncio.create_task(node.start())
        node_tasks.append(task)
        await asyncio.sleep(0.1)  # Stagger startup
    
    # Connect nodes to each other
    await asyncio.sleep(1)  # Let servers start
    
    # Create network topology
    await nodes[0].connect_to_peer('localhost', 8002)
    await nodes[1].connect_to_peer('localhost', 8003)
    await nodes[2].connect_to_peer('localhost', 8001)
    
    print("\nNetwork topology established")
    
    # Let network stabilize
    await asyncio.sleep(10)
    
    # Display network state
    for node in nodes:
        node.display_network_state()
    
    # Run for a while
    print(f"\nRunning network for 30 seconds...")
    await asyncio.sleep(30)
    
    # Final state
    print(f"\nFinal network state:")
    for node in nodes:
        node.display_network_state()
    
    # Cleanup
    for node in nodes:
        await node.stop()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_network())
```

## Testing Distributed Network

### Running Multiple Nodes

```bash
# Terminal 1
python stage6_network_blockchain.py

# The script runs all nodes in one process for simplicity
# In production, each node would run in separate processes/machines
```

### Network Monitoring

```python
async def monitor_network(nodes: List[NetworkNode], duration: int):
    """Monitor network activity for specified duration."""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        print(f"\n--- Network Status at {time.time() - start_time:.1f}s ---")
        
        for i, node in enumerate(nodes):
            canonical_chain = node.blockchain.fork_manager.get_canonical_chain()
            print(f"Node {i}: Slot {node.blockchain.current_slot}, "
                  f"Chain length {len(canonical_chain)}, "
                  f"Peers {len(node.peers)}")
        
        await asyncio.sleep(5)
```

## Key Insights

1. **Asynchronous Architecture**: `asyncio` enables handling many connections efficiently
2. **Gossip Efficiency**: Random peer selection spreads information quickly
3. **Parallel Propagation**: Turbine sharding reduces block transmission time
4. **Predictive Routing**: Gulf Stream reduces transaction latency
5. **Fault Tolerance**: Network continues operating with node failures

## Common Issues and Solutions

### Issue: "Nodes can't connect to each other"
**Problem**: Firewall or port conflicts.
**Solution**: Use different ports and check network connectivity.

### Issue: "Gossip messages not propagating"
**Problem**: Serialization errors or network partitions.
**Solution**: Add error handling and connection retry logic.

### Issue: "Block reconstruction fails"
**Problem**: Missing chunks or corruption.
**Solution**: Add timeout and retransmission mechanisms.

### Issue: "Network splits into partitions"
**Problem**: Poor connectivity or node failures.
**Solution**: This is realistic - consensus will resolve when connectivity returns.

## Extensions and Experiments

### 1. Network Topology Visualization
```python
def visualize_network_topology(nodes: List[NetworkNode]):
    """Create visual representation of network connections."""
    print("Network Topology:")
    for i, node in enumerate(nodes):
        connections = [f"Node{j}" for j, other in enumerate(nodes) 
                      if ('localhost', other.port) in node.peers]
        print(f"  Node{i} -> {', '.join(connections) if connections else 'None'}")
```

### 2. Network Performance Metrics
```python
class NetworkMetrics:
    def __init__(self):
        self.message_counts: Dict[str, int] = {}
        self.latencies: List[float] = []
        self.bandwidth_usage = 0
    
    def record_message(self, msg_type: str, size: int, latency: float):
        self.message_counts[msg_type] = self.message_counts.get(msg_type, 0) + 1
        self.latencies.append(latency)
        self.bandwidth_usage += size
    
    def get_stats(self) -> Dict:
        return {
            'total_messages': sum(self.message_counts.values()),
            'avg_latency': sum(self.latencies) / len(self.latencies) if self.latencies else 0,
            'total_bandwidth': self.bandwidth_usage,
            'message_breakdown': self.message_counts
        }
```

### 3. Dynamic Peer Discovery
```python
class PeerDiscovery:
    def __init__(self, node: NetworkNode):
        self.node = node
        self.bootstrap_peers = [('localhost', 8001)]  # Known bootstrap nodes
    
    async def discover_peers(self):
        """Actively discover new peers in the network."""
        for host, port in self.bootstrap_peers:
            await self.node.connect_to_peer(host, port)
        
        # Request peer lists from known peers
        message = NetworkMessage(
            type="peer_request",
            sender_id=self.node.node_id,
            data={},
            timestamp=time.time()
        )
        
        for host, port in list(self.node.peers):
            await self.node._send_message(host, port, message)
```

## Next Steps

You now have a fully distributed blockchain network:
- Peer-to-peer networking with gossip protocol
- Efficient block propagation using Turbine-style sharding
- Transaction forwarding with Gulf Stream optimization
- Byzantine fault tolerance across network partitions

In [Stage 7](stage-07.md), we'll add **Basic Smart Contracts**, enabling programmable logic and on-chain computation.

---

**Checkpoint Questions:**
1. How does gossip protocol ensure information reaches all nodes?
2. Why is Turbine block sharding more efficient than broadcasting whole blocks?
3. What happens when the network splits into two partitions?

**Ready for Stage 7?** → [Basic Smart Contracts](stage-07.md) 