"""Tests for the faucet ledger's atomicity guarantees."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from store import FaucetStore, current_day  # noqa: E402

DAY = "2026-08-28"
LIMIT = 1000


@pytest.fixture
async def store(tmp_path: Path):
    """Provide an opened store backed by a temporary database."""
    inst = FaucetStore(tmp_path / "faucet.db")
    await inst.open()
    yield inst
    await inst.close()


async def _credit(store: FaucetStore, user_key: str, amount: int) -> None:
    """Give a user a starting balance by way of dated check-ins."""
    for i in range(amount):
        await store.try_checkin(user_key, "tester", f"seed-{i}", 1)


@pytest.mark.asyncio
async def test_checkin_is_once_per_day(store: FaucetStore):
    ok, balance = await store.try_checkin("p:u1", "tester", DAY, 3)
    assert ok is True
    assert balance == 3

    ok, balance = await store.try_checkin("p:u1", "tester", DAY, 3)
    assert ok is False, "second check-in on the same day must be rejected"
    assert balance == 3, "rejected check-in must not change the balance"

    ok, balance = await store.try_checkin("p:u1", "tester", "2026-08-29", 2)
    assert ok is True
    assert balance == 5


@pytest.mark.asyncio
async def test_concurrent_checkin_credits_once(store: FaucetStore):
    """Ten simultaneous /签到 calls may only credit one reward."""
    results = await asyncio.gather(
        *[store.try_checkin("p:u1", "tester", DAY, 3) for _ in range(10)]
    )
    granted = [ok for ok, _ in results if ok]
    assert len(granted) == 1, f"expected exactly 1 credit, got {len(granted)}"

    user = await store.get_user("p:u1")
    assert user.balance == 3


@pytest.mark.asyncio
async def test_withdraw_rejects_insufficient_balance(store: FaucetStore):
    await _credit(store, "p:u1", 5)

    wid, reason, value = await store.reserve_withdrawal(
        "p:u1", "0xabc", 10, DAY, LIMIT
    )
    assert wid is None
    assert reason == "insufficient_balance"
    assert value == 5


@pytest.mark.asyncio
async def test_withdraw_debits_and_refund_restores(store: FaucetStore):
    await _credit(store, "p:u1", 10)

    wid, reason, _ = await store.reserve_withdrawal("p:u1", "0xabc", 4, DAY, LIMIT)
    assert wid is not None and reason == "ok"

    user = await store.get_user("p:u1")
    assert user.balance == 6, "reservation must debit immediately"
    assert await store.daily_used(DAY) == 4

    await store.refund(wid, "rpc down")
    user = await store.get_user("p:u1")
    assert user.balance == 10, "refund must restore the balance"
    assert await store.daily_used(DAY) == 0, "refund must release the quota"


@pytest.mark.asyncio
async def test_double_refund_does_not_mint(store: FaucetStore):
    await _credit(store, "p:u1", 10)
    wid, _, _ = await store.reserve_withdrawal("p:u1", "0xabc", 4, DAY, LIMIT)

    await store.refund(wid, "first")
    await store.refund(wid, "second")

    user = await store.get_user("p:u1")
    assert user.balance == 10, "a second refund must be a no-op"


@pytest.mark.asyncio
async def test_confirm_then_refund_is_rejected(store: FaucetStore):
    await _credit(store, "p:u1", 10)
    wid, _, _ = await store.reserve_withdrawal("p:u1", "0xabc", 4, DAY, LIMIT)

    await store.mark_sent(wid, "0xdeadbeef")
    await store.mark_confirmed(wid)
    await store.refund(wid, "late failure")

    user = await store.get_user("p:u1")
    assert user.balance == 6, "a confirmed withdrawal must never be refunded"
    assert user.total_withdrawn == 4


@pytest.mark.asyncio
async def test_daily_limit_holds_under_concurrency(store: FaucetStore):
    """Twenty simultaneous 100-token withdrawals against a 1000 cap.

    Exactly ten may succeed. This is the case that breaks if the quota check
    and the debit are not inside one transaction.
    """
    for i in range(20):
        await _credit(store, f"p:u{i}", 100)

    results = await asyncio.gather(
        *[
            store.reserve_withdrawal(f"p:u{i}", "0xabc", 100, DAY, LIMIT)
            for i in range(20)
        ]
    )
    approved = [wid for wid, _, _ in results if wid is not None]
    assert len(approved) == 10, f"expected 10 approvals, got {len(approved)}"
    assert await store.daily_used(DAY) == LIMIT

    # Use a funded user who was not part of the race, so the rejection can
    # only come from the quota and not from an empty balance.
    await _credit(store, "p:fresh", 50)
    wid, reason, remaining = await store.reserve_withdrawal(
        "p:fresh", "0xabc", 1, DAY, LIMIT
    )
    assert wid is None
    assert reason == "daily_limit"
    assert remaining == 0


@pytest.mark.asyncio
async def test_quota_is_per_day(store: FaucetStore):
    await _credit(store, "p:u1", 200)
    await store.reserve_withdrawal("p:u1", "0xabc", 100, DAY, LIMIT)

    assert await store.daily_used(DAY) == 100
    assert await store.daily_used("2026-08-29") == 0, "quota must reset per day"


@pytest.mark.asyncio
async def test_stale_pending_is_recoverable(store: FaucetStore):
    await _credit(store, "p:u1", 10)
    wid, _, _ = await store.reserve_withdrawal("p:u1", "0xabc", 4, DAY, LIMIT)

    # A negative age pushes the cutoff into the future, which selects the row
    # that was just written; age 0 would land on the same whole second as
    # `updated_at` and miss it.
    assert await store.find_stale_pending(max_age_seconds=-1) == [wid]
    assert await store.find_stale_pending(max_age_seconds=3600) == []

    await store.mark_sent(wid, "0xdeadbeef")
    assert await store.find_stale_pending(max_age_seconds=-1) == [], (
        "a broadcast withdrawal is no longer pending and must not be refunded"
    )


@pytest.mark.asyncio
async def test_transfer_moves_balance_and_creates_recipient(store: FaucetStore):
    await _credit(store, "p:alice", 10)

    ok, remaining = await store.transfer_balance("p:alice", "p:bob", 4)
    assert ok is True
    assert remaining == 6

    assert (await store.get_user("p:alice")).balance == 6
    bob = await store.get_user("p:bob")
    assert bob.balance == 4, "recipient must be created on first transfer"


@pytest.mark.asyncio
async def test_transfer_rejects_insufficient_balance(store: FaucetStore):
    await _credit(store, "p:alice", 3)

    ok, balance = await store.transfer_balance("p:alice", "p:bob", 5)
    assert ok is False
    assert balance == 3
    assert (await store.get_user("p:bob")).balance == 0


@pytest.mark.asyncio
async def test_concurrent_transfers_never_overdraw(store: FaucetStore):
    """Ten simultaneous 30-token transfers from a 100 balance: exactly 3 pass."""
    await _credit(store, "p:alice", 100)

    results = await asyncio.gather(
        *[store.transfer_balance("p:alice", f"p:u{i}", 30) for i in range(10)]
    )
    succeeded = [ok for ok, _ in results if ok]
    assert len(succeeded) == 3, f"expected 3 transfers, got {len(succeeded)}"
    assert (await store.get_user("p:alice")).balance == 10

    received = 0
    for i in range(10):
        received += (await store.get_user(f"p:u{i}")).balance
    assert received == 90, "credited total must equal debited total"


@pytest.mark.asyncio
async def test_wallet_binding_roundtrip(store: FaucetStore):
    await store.set_wallet("p:u1", "0x1234")
    assert (await store.get_user("p:u1")).wallet == "0x1234"

    await store.set_wallet("p:u1", None)
    assert (await store.get_user("p:u1")).wallet is None


@pytest.mark.asyncio
async def test_list_bound_wallets(store: FaucetStore):
    await store.set_wallet("111", "0xaaaa")
    await store.set_wallet("222", "0xbbbb")
    await _credit(store, "333", 5)  # has balance but no wallet

    bound = sorted(await store.list_bound_wallets())
    assert bound == [("111", "0xaaaa"), ("222", "0xbbbb")]

    await store.set_wallet("111", None)
    assert await store.list_bound_wallets() == [("222", "0xbbbb")]


@pytest.mark.asyncio
async def test_grant_credits_balance_and_counts_as_earned(store: FaucetStore):
    granted, balance = await store.grant("p:u1", "tester", 7, "soupai", "round", DAY)
    assert (granted, balance) == (7, 7)

    user = await store.get_user("p:u1")
    assert user.total_earned == 7
    assert user.checkin_count == 0, "a grant is not a check-in"


@pytest.mark.asyncio
async def test_grant_is_clipped_by_the_daily_cap(store: FaucetStore):
    granted, _ = await store.grant("p:u1", "t", 30, "soupai", "r", DAY, daily_cap=50)
    assert granted == 30

    # The second round only gets what is left of the cap, not the full ask.
    granted, balance = await store.grant(
        "p:u1", "t", 30, "soupai", "r", DAY, daily_cap=50
    )
    assert (granted, balance) == (20, 50)

    granted, _ = await store.grant("p:u1", "t", 5, "soupai", "r", DAY, daily_cap=50)
    assert granted == 0


@pytest.mark.asyncio
async def test_grant_cap_is_scoped_to_source_and_day(store: FaucetStore):
    await store.grant("p:u1", "t", 50, "soupai", "r", DAY, daily_cap=50)

    other_game, _ = await store.grant("p:u1", "t", 10, "quiz", "r", DAY, daily_cap=50)
    assert other_game == 10, "another plugin has its own allowance"

    tomorrow, _ = await store.grant(
        "p:u1", "t", 10, "soupai", "r", "2026-08-29", daily_cap=50
    )
    assert tomorrow == 10, "the cap resets with the faucet day"


@pytest.mark.asyncio
async def test_grant_ignores_non_positive_amounts(store: FaucetStore):
    await _credit(store, "p:u1", 4)

    assert await store.grant("p:u1", "t", 0, "soupai", "r", DAY) == (0, 4)
    assert await store.grant("p:u1", "t", -5, "soupai", "r", DAY) == (0, 4)


@pytest.mark.asyncio
async def test_grant_keeps_the_stored_name_when_none_is_supplied(store: FaucetStore):
    await store.grant("p:u1", "Player One", 3, "soupai", "r", DAY)
    await store.grant("p:u1", "", 3, "soupai", "r", DAY)

    cursor = await store._conn.execute(
        "SELECT display_name FROM users WHERE user_key = ?", ("p:u1",)
    )
    row = await cursor.fetchone()
    await cursor.close()
    assert row["display_name"] == "Player One"


@pytest.mark.asyncio
async def test_concurrent_grants_never_exceed_the_cap(store: FaucetStore):
    """Ten simultaneous 10-token grants against a 50 cap.

    This is the case that breaks if the cap lookup and the insert are not
    inside one transaction: every caller would read the same stale total.
    """
    results = await asyncio.gather(
        *[
            store.grant("p:u1", "t", 10, "soupai", "r", DAY, daily_cap=50)
            for _ in range(10)
        ]
    )

    assert sum(granted for granted, _ in results) == 50
    assert (await store.get_user("p:u1")).balance == 50


def test_current_day_uses_offset():
    """The day string must follow the configured offset, not the host clock."""
    assert len(current_day(8)) == 10
    assert current_day(8) >= current_day(-12)


@pytest.mark.asyncio
async def test_api_user_binding_roundtrip(store: FaucetStore):
    await store.set_api_user("p:u1", 147)
    assert (await store.get_user("p:u1")).api_user_id == 147

    await store.set_api_user("p:u1", 233)
    assert (await store.get_user("p:u1")).api_user_id == 233, (
        "rebinding must overwrite the previous id"
    )

    await store.set_api_user("p:u1", None)
    assert (await store.get_user("p:u1")).api_user_id is None


@pytest.mark.asyncio
async def test_recharge_rejects_insufficient_balance(store: FaucetStore):
    await _credit(store, "p:u1", 5)

    rid, reason, value = await store.reserve_recharge(
        "p:u1", 147, 10, 5_000_000, DAY, 500
    )
    assert rid is None
    assert reason == "insufficient_balance"
    assert value == 5
    assert (await store.get_user("p:u1")).balance == 5


@pytest.mark.asyncio
async def test_recharge_debits_and_refund_restores(store: FaucetStore):
    await _credit(store, "p:u1", 10)

    rid, reason, remaining = await store.reserve_recharge(
        "p:u1", 147, 4, 2_000_000, DAY, 500
    )
    assert rid is not None and reason == "ok"
    assert remaining == 496
    assert (await store.get_user("p:u1")).balance == 6, (
        "reservation must debit immediately"
    )

    await store.mark_recharged(rid)
    assert (await store.get_user("p:u1")).balance == 6, "a top-up is final"

    rid2, _, _ = await store.reserve_recharge("p:u1", 147, 3, 1_500_000, DAY, 500)
    await store.refund_recharge(rid2, "api down")
    assert (await store.get_user("p:u1")).balance == 6, (
        "refund must restore the coins"
    )
    assert await store.daily_recharge_used(DAY) == 4, (
        "refund must release the daily quota again"
    )


@pytest.mark.asyncio
async def test_double_recharge_refund_does_not_mint(store: FaucetStore):
    await _credit(store, "p:u1", 10)
    rid, _, _ = await store.reserve_recharge("p:u1", 147, 4, 2_000_000, DAY, 500)

    await store.refund_recharge(rid, "first")
    await store.refund_recharge(rid, "second")

    assert (await store.get_user("p:u1")).balance == 10, (
        "a second refund must be a no-op"
    )


@pytest.mark.asyncio
async def test_confirmed_recharge_cannot_be_refunded(store: FaucetStore):
    await _credit(store, "p:u1", 10)
    rid, _, _ = await store.reserve_recharge("p:u1", 147, 4, 2_000_000, DAY, 500)

    await store.mark_recharged(rid)
    await store.refund_recharge(rid, "late failure")

    assert (await store.get_user("p:u1")).balance == 6, (
        "a successful recharge must never be refunded"
    )


@pytest.mark.asyncio
async def test_concurrent_recharges_never_overdraw(store: FaucetStore):
    """Ten simultaneous 30-coin recharges from a 100 balance: exactly 3 pass."""
    await _credit(store, "p:alice", 100)

    results = await asyncio.gather(
        *[
            store.reserve_recharge("p:alice", 147, 30, 15_000_000, DAY, 500)
            for _ in range(10)
        ]
    )
    approved = [rid for rid, _, _ in results if rid is not None]
    assert len(approved) == 3, f"expected 3 approvals, got {len(approved)}"
    assert (await store.get_user("p:alice")).balance == 10


@pytest.mark.asyncio
async def test_recharge_daily_limit_holds_under_concurrency(store: FaucetStore):
    """Ten simultaneous 100-coin recharges against a 500 cap: exactly 5 pass."""
    for i in range(10):
        await _credit(store, f"p:u{i}", 100)

    results = await asyncio.gather(
        *[
            store.reserve_recharge(f"p:u{i}", 147, 100, 50_000_000, DAY, 500)
            for i in range(10)
        ]
    )
    approved = [rid for rid, _, _ in results if rid is not None]
    assert len(approved) == 5, f"expected 5 approvals, got {len(approved)}"
    assert await store.daily_recharge_used(DAY) == 500

    # A funded user outside the race can only be rejected by the quota.
    await _credit(store, "p:fresh", 50)
    rid, reason, remaining = await store.reserve_recharge(
        "p:fresh", 147, 1, 500_000, DAY, 500
    )
    assert rid is None
    assert reason == "daily_limit"
    assert remaining == 0


@pytest.mark.asyncio
async def test_recharge_quota_is_per_day(store: FaucetStore):
    await _credit(store, "p:u1", 200)
    await store.reserve_recharge("p:u1", 147, 100, 50_000_000, DAY, 500)

    assert await store.daily_recharge_used(DAY) == 100
    assert await store.daily_recharge_used("2026-08-29") == 0, (
        "the recharge limit must reset per day"
    )
    rid, reason, _ = await store.reserve_recharge(
        "p:u1", 147, 100, 50_000_000, "2026-08-29", 500
    )
    assert rid is not None and reason == "ok", "the next day gets a fresh cap"


@pytest.mark.asyncio
async def test_stale_pending_recharge_is_recoverable(store: FaucetStore):
    await _credit(store, "p:u1", 10)
    rid, _, _ = await store.reserve_recharge("p:u1", 147, 4, 2_000_000, DAY, 500)

    assert await store.find_stale_pending_recharges(max_age_seconds=-1) == [rid]
    assert await store.find_stale_pending_recharges(max_age_seconds=3600) == []

    await store.mark_recharged(rid)
    assert await store.find_stale_pending_recharges(max_age_seconds=-1) == [], (
        "a finished recharge is no longer pending and must not be refunded"
    )
