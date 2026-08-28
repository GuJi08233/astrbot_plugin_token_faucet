"""Async ERC-20 transfer client for the token faucet plugin.

All amounts are raw on-chain integers. The faucet token has ``decimals == 0``,
so one token is literally ``1`` on chain and no scaling is applied here; the
configured ``decimals`` is still honoured for other tokens.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

try:
    from web3 import AsyncHTTPProvider, AsyncWeb3, Web3

    WEB3_IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on install state
    AsyncHTTPProvider = AsyncWeb3 = Web3 = None  # type: ignore[assignment]
    WEB3_IMPORT_ERROR = str(exc)

# Minimal ERC-20 surface: the faucet only reads balances/metadata and transfers.
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


class ChainError(Exception):
    """Raised when an on-chain operation cannot be completed."""


@dataclass
class TransferResult:
    """Outcome of a broadcast ERC-20 transfer."""

    tx_hash: str
    nonce: int
    confirmed: bool = False
    gas_used: int | None = None


class TokenChainClient:
    """Sends ERC-20 transfers from a single hot wallet.

    Every send is serialized behind an ``asyncio.Lock``. That serialization is
    required rather than cosmetic: two concurrent transfers would otherwise
    fetch the same ``pending`` nonce and the second transaction would replace
    the first instead of being mined alongside it.
    """

    def __init__(
        self,
        rpc_url: str,
        private_key: str,
        token_address: str,
        chain_id: int,
        gas_limit: int = 120000,
        request_timeout: float = 30.0,
        decimals: int = 0,
    ) -> None:
        if WEB3_IMPORT_ERROR is not None:
            raise ChainError(
                f"web3 is not installed ({WEB3_IMPORT_ERROR}). "
                "Install it with: pip install web3",
            )
        if not rpc_url:
            raise ChainError("RPC URL is not configured")
        if not private_key:
            raise ChainError("Private key is not configured")
        if not token_address:
            raise ChainError("Token contract address is not configured")

        key = private_key.strip()
        if not key.startswith("0x"):
            key = "0x" + key
        try:
            self._account = Web3().eth.account.from_key(key)
        except Exception as exc:
            raise ChainError(f"Invalid private key: {exc}") from exc

        self._w3 = AsyncWeb3(
            AsyncHTTPProvider(rpc_url, request_kwargs={"timeout": request_timeout}),
        )
        self._token_address = Web3.to_checksum_address(token_address)
        self._contract = self._w3.eth.contract(
            address=self._token_address,
            abi=ERC20_ABI,
        )
        self._chain_id = chain_id
        self._gas_limit = gas_limit
        self._decimals = decimals
        self._send_lock = asyncio.Lock()
        # Locally tracked next nonce. The node's `pending` count lags behind
        # rapid back-to-back sends, so we keep our own high-water mark.
        self._local_nonce: int | None = None

    @property
    def address(self) -> str:
        """Return the hot wallet's checksummed address."""
        return self._account.address

    @property
    def token_address(self) -> str:
        """Return the token contract's checksummed address."""
        return self._token_address

    @staticmethod
    def is_valid_address(address: str) -> bool:
        """Check whether a string is a well-formed Ethereum address.

        Args:
            address: Candidate address, with or without checksum casing.

        Returns:
            ``True`` when the address is structurally valid.
        """
        if WEB3_IMPORT_ERROR is not None or not address:
            return False
        return bool(Web3.is_address(address))

    @staticmethod
    def to_checksum(address: str) -> str:
        """Normalise an address to EIP-55 checksum form.

        Args:
            address: A structurally valid Ethereum address.

        Returns:
            The checksummed address.
        """
        return Web3.to_checksum_address(address)

    def to_raw_amount(self, amount: int) -> int:
        """Convert a token count to the contract's raw integer unit.

        Args:
            amount: Human-facing token count.

        Returns:
            The value to pass to ``transfer``.
        """
        return amount * (10**self._decimals)

    async def token_balance(self, address: str | None = None) -> int:
        """Read an address's token balance in human units.

        Args:
            address: Address to query; defaults to the hot wallet.

        Returns:
            The balance divided down by the token's decimals.

        Raises:
            ChainError: If the RPC call fails.
        """
        target = Web3.to_checksum_address(address) if address else self.address
        try:
            raw = await self._contract.functions.balanceOf(target).call()
        except Exception as exc:
            raise ChainError(f"Failed to read token balance: {exc}") from exc
        return int(raw) // (10**self._decimals)

    async def token_info(self) -> dict:
        """Read the token's descriptive fields from the contract.

        The total supply is scaled by the contract's own reported decimals
        rather than the configured value, so the display always matches what
        the chain says even if the plugin config is stale.

        Returns:
            Mapping with ``name``, ``symbol``, ``decimals`` and ``total_supply``
            (in human units).

        Raises:
            ChainError: If any call fails.
        """
        try:
            name = await self._contract.functions.name().call()
            symbol = await self._contract.functions.symbol().call()
            decimals = int(await self._contract.functions.decimals().call())
            raw_supply = await self._contract.functions.totalSupply().call()
        except Exception as exc:
            raise ChainError(f"Failed to read token info: {exc}") from exc
        return {
            "name": str(name),
            "symbol": str(symbol),
            "decimals": decimals,
            "total_supply": int(raw_supply) // (10**decimals),
        }

    async def transfer(
        self,
        to_address: str,
        amount: int,
        wait_for_receipt: bool = True,
        receipt_timeout: float = 180.0,
    ) -> TransferResult:
        """Transfer tokens from the hot wallet to a recipient.

        Args:
            to_address: Checksummed destination address.
            amount: Token count in human units.
            wait_for_receipt: Whether to await mining before returning.
            receipt_timeout: Seconds to wait for the receipt.

        Returns:
            The broadcast result. ``confirmed`` is ``False`` when the receipt
            was not awaited or did not arrive in time - the transaction may
            still be mined later.

        Raises:
            ChainError: If the transfer could not be broadcast, or if it was
                mined with a reverted status.
        """
        recipient = Web3.to_checksum_address(to_address)
        raw_amount = self.to_raw_amount(amount)

        async with self._send_lock:
            try:
                chain_nonce = await self._w3.eth.get_transaction_count(
                    self.address,
                    "pending",
                )
            except Exception as exc:
                raise ChainError(f"Failed to fetch nonce: {exc}") from exc

            nonce = chain_nonce
            if self._local_nonce is not None and self._local_nonce > chain_nonce:
                nonce = self._local_nonce

            try:
                tx = await self._contract.functions.transfer(
                    recipient,
                    raw_amount,
                ).build_transaction(
                    {
                        "from": self.address,
                        "nonce": nonce,
                        "chainId": self._chain_id,
                        "gas": self._gas_limit,
                    },
                )
            except Exception as exc:
                # A revert during gas estimation lands here, which is the usual
                # symptom of the hot wallet holding too few tokens.
                raise ChainError(f"Failed to build transfer: {exc}") from exc

            try:
                signed = self._w3.eth.account.sign_transaction(
                    tx,
                    self._account.key,
                )
                # web3 v7 renamed `rawTransaction` to `raw_transaction`.
                raw_tx = getattr(signed, "raw_transaction", None)
                if raw_tx is None:
                    raw_tx = signed.rawTransaction
                tx_hash = await self._w3.eth.send_raw_transaction(raw_tx)
            except Exception as exc:
                raise ChainError(f"Failed to broadcast transfer: {exc}") from exc

            self._local_nonce = nonce + 1

        tx_hash_hex = tx_hash.hex()
        if not tx_hash_hex.startswith("0x"):
            tx_hash_hex = "0x" + tx_hash_hex
        result = TransferResult(tx_hash=tx_hash_hex, nonce=nonce)

        if not wait_for_receipt:
            return result

        try:
            receipt = await self._w3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=receipt_timeout,
            )
        except asyncio.TimeoutError:
            # Broadcast succeeded but mining is slow. The caller keeps the
            # withdrawal as `sent` rather than refunding, because the transfer
            # may still land.
            return result
        except Exception:
            return result

        result.gas_used = int(receipt.get("gasUsed", 0)) or None
        if int(receipt.get("status", 0)) != 1:
            raise ChainError(f"Transaction reverted on chain: {tx_hash_hex}")
        result.confirmed = True
        return result

    async def wait_for_receipt(
        self,
        tx_hash: str,
        timeout: float = 180.0,
    ) -> tuple[bool, int | None]:
        """Wait for a broadcast transaction to be mined.

        Args:
            tx_hash: The ``0x``-prefixed transaction hash.
            timeout: Seconds to wait before giving up.

        Returns:
            ``(succeeded, gas_used)``. ``succeeded`` is ``False`` when the
            transaction reverted.

        Raises:
            ChainError: If the receipt did not arrive within the timeout, which
                leaves the transaction's fate undecided.
        """
        try:
            receipt = await self._w3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=timeout,
            )
        except Exception as exc:
            raise ChainError(f"Receipt not available: {exc}") from exc
        gas_used = int(receipt.get("gasUsed", 0)) or None
        return int(receipt.get("status", 0)) == 1, gas_used

    async def close(self) -> None:
        """Release the underlying HTTP session if the provider owns one."""
        provider = getattr(self._w3, "provider", None)
        disconnect = getattr(provider, "disconnect", None)
        if disconnect is None:
            return
        try:
            await disconnect()
        except Exception:
            # Shutdown path: a failed session close must not mask the real
            # termination flow.
            pass
