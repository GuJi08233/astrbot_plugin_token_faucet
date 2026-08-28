"""SQLite persistence for the token faucet plugin.

The token has ``decimals == 0``, so every amount in this module is a plain
integer number of tokens - there is no fixed-point conversion anywhere.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

# Withdrawal states that still occupy the daily global quota. A failed or
# refunded withdrawal releases its quota again so a broken RPC call does not
# silently burn the day's budget.
QUOTA_STATUSES = ("pending", "sent", "confirmed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_key         TEXT PRIMARY KEY,
    display_name     TEXT,
    balance          INTEGER NOT NULL DEFAULT 0,
    wallet           TEXT,
    last_checkin_day TEXT,
    checkin_count    INTEGER NOT NULL DEFAULT 0,
    total_earned     INTEGER NOT NULL DEFAULT 0,
    total_withdrawn  INTEGER NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS withdrawals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key    TEXT NOT NULL,
    to_address  TEXT NOT NULL,
    amount      INTEGER NOT NULL,
    day         TEXT NOT NULL,
    status      TEXT NOT NULL,
    tx_hash     TEXT,
    error       TEXT,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_withdrawals_day ON withdrawals (day, status);
CREATE INDEX IF NOT EXISTS idx_withdrawals_user ON withdrawals (user_key, id DESC);

CREATE TABLE IF NOT EXISTS checkins (
    user_key TEXT NOT NULL,
    day      TEXT NOT NULL,
    amount   INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_key, day)
);
"""


def current_day(utc_offset_hours: int) -> str:
    """Return the faucet's current calendar day as ``YYYY-MM-DD``.

    A fixed UTC offset is used instead of ``zoneinfo`` so the plugin does not
    depend on the ``tzdata`` package being present, which is not guaranteed on
    slim Linux images or Windows.

    Args:
        utc_offset_hours: Offset from UTC that defines when the day rolls over.

    Returns:
        The current day string in the configured offset.
    """
    tz = timezone(timedelta(hours=utc_offset_hours))
    return datetime.now(tz).strftime("%Y-%m-%d")


@dataclass
class UserRecord:
    """A faucet user's persisted state."""

    user_key: str
    balance: int = 0
    wallet: str | None = None
    last_checkin_day: str | None = None
    checkin_count: int = 0
    total_earned: int = 0
    total_withdrawn: int = 0


class FaucetStore:
    """Async SQLite store guarding balances, wallets and withdrawal quota.

    A single connection is reused and every write path is serialized behind an
    ``asyncio.Lock``. Combined with ``BEGIN IMMEDIATE`` transactions this keeps
    the balance debit and the daily-quota check atomic, which is what stops
    concurrent ``/提现`` commands from overdrawing either budget.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        """Open the database and apply the schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        # WAL keeps readers from blocking the writer; NORMAL is durable enough
        # for a faucet ledger and avoids an fsync on every statement.
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection if it is open."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("FaucetStore is not opened")
        return self._db

    async def get_user(self, user_key: str) -> UserRecord:
        """Return a user's record, or an empty record when unknown.

        Args:
            user_key: Stable sender identifier.

        Returns:
            The stored record, or a zeroed record for first-time users.
        """
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT * FROM users WHERE user_key = ?",
                (user_key,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return UserRecord(user_key=user_key)
        return UserRecord(
            user_key=row["user_key"],
            balance=row["balance"],
            wallet=row["wallet"],
            last_checkin_day=row["last_checkin_day"],
            checkin_count=row["checkin_count"],
            total_earned=row["total_earned"],
            total_withdrawn=row["total_withdrawn"],
        )

    async def try_checkin(
        self,
        user_key: str,
        display_name: str,
        day: str,
        amount: int,
    ) -> tuple[bool, int]:
        """Credit a daily check-in exactly once per day.

        The ``checkins`` primary key ``(user_key, day)`` is the real guard: a
        second call on the same day hits the unique constraint instead of
        relying on a read-then-write race.

        Args:
            user_key: Stable sender identifier.
            display_name: Latest known nickname, refreshed on every check-in.
            day: Faucet day string from :func:`current_day`.
            amount: Number of tokens to credit.

        Returns:
            ``(True, new_balance)`` on success, or ``(False, current_balance)``
            when the user already checked in today.
        """
        now = int(time.time())
        async with self._lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._conn.execute(
                    "SELECT 1 FROM checkins WHERE user_key = ? AND day = ?",
                    (user_key, day),
                )
                already = await cursor.fetchone()
                await cursor.close()
                if already is not None:
                    cursor = await self._conn.execute(
                        "SELECT balance FROM users WHERE user_key = ?",
                        (user_key,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                    await self._conn.rollback()
                    return False, int(row["balance"]) if row else 0

                await self._conn.execute(
                    "INSERT INTO checkins (user_key, day, amount, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (user_key, day, amount, now),
                )
                await self._conn.execute(
                    """
                    INSERT INTO users (
                        user_key, display_name, balance, last_checkin_day,
                        checkin_count, total_earned, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(user_key) DO UPDATE SET
                        display_name = excluded.display_name,
                        balance = users.balance + excluded.balance,
                        last_checkin_day = excluded.last_checkin_day,
                        checkin_count = users.checkin_count + 1,
                        total_earned = users.total_earned + excluded.total_earned,
                        updated_at = excluded.updated_at
                    """,
                    (user_key, display_name, amount, day, amount, now, now),
                )
                cursor = await self._conn.execute(
                    "SELECT balance FROM users WHERE user_key = ?",
                    (user_key,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return True, int(row["balance"]) if row else amount

    async def set_wallet(self, user_key: str, wallet: str | None) -> None:
        """Bind or clear a user's payout address.

        Args:
            user_key: Stable sender identifier.
            wallet: Checksummed address, or ``None`` to unbind.
        """
        now = int(time.time())
        async with self._lock:
            await self._conn.execute(
                """
                INSERT INTO users (user_key, wallet, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_key) DO UPDATE SET
                    wallet = excluded.wallet,
                    updated_at = excluded.updated_at
                """,
                (user_key, wallet, now, now),
            )
            await self._conn.commit()

    async def list_bound_wallets(self) -> list[tuple[str, str]]:
        """Return every user that has a bound wallet.

        Returns:
            ``(user_key, wallet)`` tuples, one per bound user.
        """
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT user_key, wallet FROM users WHERE wallet IS NOT NULL",
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [(r["user_key"], r["wallet"]) for r in rows]

    async def transfer_balance(
        self,
        from_key: str,
        to_key: str,
        amount: int,
    ) -> tuple[bool, int]:
        """Move plugin balance between two users atomically.

        Purely an in-ledger move: no chain interaction and no effect on the
        daily withdrawal quota. The recipient row is created on the fly so a
        user can receive tokens before their first check-in.

        Args:
            from_key: Sender's stable identifier.
            to_key: Recipient's stable identifier.
            amount: Number of tokens to move; must be positive.

        Returns:
            ``(True, sender_new_balance)`` on success, or
            ``(False, sender_current_balance)`` when the sender's balance is
            insufficient.
        """
        now = int(time.time())
        async with self._lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._conn.execute(
                    "SELECT balance FROM users WHERE user_key = ?",
                    (from_key,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                balance = int(row["balance"]) if row else 0
                if balance < amount:
                    await self._conn.rollback()
                    return False, balance

                await self._conn.execute(
                    "UPDATE users SET balance = balance - ?, updated_at = ? "
                    "WHERE user_key = ?",
                    (amount, now, from_key),
                )
                await self._conn.execute(
                    """
                    INSERT INTO users (user_key, balance, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_key) DO UPDATE SET
                        balance = users.balance + excluded.balance,
                        updated_at = excluded.updated_at
                    """,
                    (to_key, amount, now, now),
                )
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return True, balance - amount

    async def daily_used(self, day: str) -> int:
        """Return how much of the global daily quota is already committed.

        Args:
            day: Faucet day string from :func:`current_day`.

        Returns:
            Sum of amounts in pending, sent and confirmed withdrawals.
        """
        placeholders = ",".join("?" * len(QUOTA_STATUSES))
        async with self._lock:
            cursor = await self._conn.execute(
                f"SELECT COALESCE(SUM(amount), 0) AS used FROM withdrawals "
                f"WHERE day = ? AND status IN ({placeholders})",
                (day, *QUOTA_STATUSES),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return int(row["used"]) if row else 0

    async def reserve_withdrawal(
        self,
        user_key: str,
        to_address: str,
        amount: int,
        day: str,
        daily_limit: int,
    ) -> tuple[int | None, str, int]:
        """Debit the user and reserve daily quota in one transaction.

        This must succeed *before* any on-chain call is made, so a crash between
        the debit and the transfer leaves a recoverable ``pending`` row rather
        than an untracked payout.

        Args:
            user_key: Stable sender identifier.
            to_address: Checksummed destination address.
            amount: Number of tokens to withdraw.
            day: Faucet day string from :func:`current_day`.
            daily_limit: Global per-day cap in tokens.

        Returns:
            ``(withdrawal_id, "ok", remaining_quota)`` on success. On rejection,
            ``(None, reason, value)`` where reason is ``"insufficient_balance"``
            (value = current balance) or ``"daily_limit"`` (value = remaining
            quota).
        """
        now = int(time.time())
        placeholders = ",".join("?" * len(QUOTA_STATUSES))
        async with self._lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._conn.execute(
                    "SELECT balance FROM users WHERE user_key = ?",
                    (user_key,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                balance = int(row["balance"]) if row else 0
                if balance < amount:
                    await self._conn.rollback()
                    return None, "insufficient_balance", balance

                cursor = await self._conn.execute(
                    f"SELECT COALESCE(SUM(amount), 0) AS used FROM withdrawals "
                    f"WHERE day = ? AND status IN ({placeholders})",
                    (day, *QUOTA_STATUSES),
                )
                quota_row = await cursor.fetchone()
                await cursor.close()
                used = int(quota_row["used"]) if quota_row else 0
                remaining = daily_limit - used
                if amount > remaining:
                    await self._conn.rollback()
                    return None, "daily_limit", max(remaining, 0)

                await self._conn.execute(
                    "UPDATE users SET balance = balance - ?, updated_at = ? "
                    "WHERE user_key = ?",
                    (amount, now, user_key),
                )
                cursor = await self._conn.execute(
                    """
                    INSERT INTO withdrawals (
                        user_key, to_address, amount, day, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (user_key, to_address, amount, day, now, now),
                )
                withdrawal_id = cursor.lastrowid
                await cursor.close()
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return withdrawal_id, "ok", remaining - amount

    async def mark_sent(self, withdrawal_id: int, tx_hash: str) -> None:
        """Record a broadcast transaction hash for a reserved withdrawal.

        Args:
            withdrawal_id: Row id returned by :meth:`reserve_withdrawal`.
            tx_hash: The ``0x``-prefixed transaction hash.
        """
        now = int(time.time())
        async with self._lock:
            await self._conn.execute(
                "UPDATE withdrawals SET status = 'sent', tx_hash = ?, "
                "updated_at = ? WHERE id = ?",
                (tx_hash, now, withdrawal_id),
            )
            await self._conn.commit()

    async def mark_confirmed(self, withdrawal_id: int) -> None:
        """Mark a withdrawal as confirmed on chain and count it as spent.

        Args:
            withdrawal_id: Row id returned by :meth:`reserve_withdrawal`.
        """
        now = int(time.time())
        async with self._lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._conn.execute(
                    "SELECT user_key, amount, status FROM withdrawals WHERE id = ?",
                    (withdrawal_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None or row["status"] == "confirmed":
                    await self._conn.rollback()
                    return
                await self._conn.execute(
                    "UPDATE withdrawals SET status = 'confirmed', updated_at = ? "
                    "WHERE id = ?",
                    (now, withdrawal_id),
                )
                await self._conn.execute(
                    "UPDATE users SET total_withdrawn = total_withdrawn + ?, "
                    "updated_at = ? WHERE user_key = ?",
                    (int(row["amount"]), now, row["user_key"]),
                )
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise

    async def refund(self, withdrawal_id: int, error: str) -> None:
        """Return a failed withdrawal's tokens to the user and free its quota.

        Marking the row ``failed`` removes it from :data:`QUOTA_STATUSES`, so the
        reserved daily quota becomes available again.

        Args:
            withdrawal_id: Row id returned by :meth:`reserve_withdrawal`.
            error: Short failure reason stored for diagnostics.
        """
        now = int(time.time())
        async with self._lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._conn.execute(
                    "SELECT user_key, amount, status FROM withdrawals WHERE id = ?",
                    (withdrawal_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                # Only a still-reserved row may be refunded; refunding twice
                # would mint tokens out of nothing.
                if row is None or row["status"] not in ("pending", "sent"):
                    await self._conn.rollback()
                    return
                await self._conn.execute(
                    "UPDATE withdrawals SET status = 'failed', error = ?, "
                    "updated_at = ? WHERE id = ?",
                    (error[:500], now, withdrawal_id),
                )
                await self._conn.execute(
                    "UPDATE users SET balance = balance + ?, updated_at = ? "
                    "WHERE user_key = ?",
                    (int(row["amount"]), now, row["user_key"]),
                )
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise

    async def list_withdrawals(
        self,
        user_key: str,
        limit: int = 5,
    ) -> list[aiosqlite.Row]:
        """Return a user's most recent withdrawals, newest first.

        Args:
            user_key: Stable sender identifier.
            limit: Maximum number of rows to return.

        Returns:
            A list of withdrawal rows.
        """
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT * FROM withdrawals WHERE user_key = ? ORDER BY id DESC LIMIT ?",
                (user_key, limit),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return list(rows)

    async def find_stale_pending(self, max_age_seconds: int = 300) -> list[int]:
        """Find withdrawals stuck in ``pending`` past a grace period.

        A row stays ``pending`` only between the debit and a successful
        broadcast, so anything older than the grace period means the process
        died mid-flight and the tokens should go back to the user.

        Args:
            max_age_seconds: Age past which a pending row is considered stale.

        Returns:
            The ids of stale pending withdrawals.
        """
        cutoff = int(time.time()) - max_age_seconds
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT id FROM withdrawals WHERE status = 'pending' "
                "AND updated_at < ?",
                (cutoff,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [int(row["id"]) for row in rows]
