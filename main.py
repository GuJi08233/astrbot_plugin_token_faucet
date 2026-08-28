"""Token faucet plugin: daily check-in rewards redeemable as ERC-20 tokens."""

from __future__ import annotations

import asyncio
import random
import re
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import At
from astrbot.api.star import Context, Star, StarTools

from .chain import ChainError, TokenChainClient
from .store import FaucetStore, current_day

# Grace period after which a withdrawal still stuck in `pending` is treated as
# a crashed send and refunded on the next startup.
STALE_PENDING_SECONDS = 300


class TokenFaucet(Star):
    """Daily check-in faucet backed by on-chain ERC-20 payouts."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self._config = config
        self._store = FaucetStore(Path(StarTools.get_data_dir()) / "faucet.db")
        self._client: TokenChainClient | None = None
        self._chain_error: str | None = None
        self._bg_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ #
    # Config accessors
    # ------------------------------------------------------------------ #

    def _chain_cfg(self) -> dict:
        return self._config.get("chain", {}) or {}

    def _faucet_cfg(self) -> dict:
        return self._config.get("faucet", {}) or {}

    def _int_cfg(self, section: dict, key: str, default: int) -> int:
        """Read an integer config value, falling back on malformed input.

        Args:
            section: The config sub-object to read from.
            key: Key to look up.
            default: Value used when missing or unparseable.

        Returns:
            The resolved integer.
        """
        try:
            value = section.get(key, default)
            return int(value) if value not in (None, "") else default
        except (TypeError, ValueError):
            return default

    @property
    def _symbol(self) -> str:
        return str(self._faucet_cfg().get("token_symbol") or "币")

    @property
    def _day(self) -> str:
        return current_day(self._int_cfg(self._faucet_cfg(), "day_utc_offset", 8))

    def _user_key(self, event: AstrMessageEvent) -> str:
        """Build a stable cross-session identity for a sender.

        The platform id is included so the same numeric user id on two
        platforms never collides, and the session/group is deliberately left
        out so a user's balance follows them across groups.

        Args:
            event: The incoming message event.

        Returns:
            A ``platform:sender_id`` key.
        """
        return f"{event.get_platform_id()}:{event.get_sender_id()}"

    def _explorer_tx(self, tx_hash: str) -> str:
        base = str(self._chain_cfg().get("explorer_base") or "").rstrip("/")
        return f"{base}/tx/{tx_hash}" if base else tx_hash

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Open the ledger, build the chain client and recover stale rows."""
        await self._store.open()

        chain = self._chain_cfg()
        try:
            self._client = TokenChainClient(
                rpc_url=str(chain.get("rpc_url", "")),
                private_key=str(chain.get("private_key", "")),
                token_address=str(chain.get("token_address", "")),
                chain_id=self._int_cfg(chain, "chain_id", 11155111),
                gas_limit=self._int_cfg(chain, "gas_limit", 120000),
                request_timeout=float(self._int_cfg(chain, "request_timeout", 30)),
                decimals=self._int_cfg(chain, "token_decimals", 0),
            )
            logger.info(f"Token faucet hot wallet: {self._client.address}")
        except ChainError as exc:
            self._chain_error = str(exc)
            logger.warning(f"Token faucet chain client unavailable: {exc}")

        # A row left in `pending` means the process died between debiting the
        # user and broadcasting, so the tokens were never sent out.
        try:
            stale = await self._store.find_stale_pending(STALE_PENDING_SECONDS)
            for withdrawal_id in stale:
                await self._store.refund(withdrawal_id, "recovered on startup")
            if stale:
                logger.warning(
                    f"Token faucet refunded {len(stale)} stale pending withdrawals",
                )
        except Exception as exc:
            logger.error(f"Token faucet startup recovery failed: {exc}")

    async def terminate(self) -> None:
        """Cancel background confirmations and close all resources."""
        for task in list(self._bg_tasks):
            task.cancel()
        self._bg_tasks.clear()
        if self._client is not None:
            await self._client.close()
        await self._store.close()

    def _spawn(self, coro) -> None:
        """Run a coroutine detached while holding a strong reference.

        Args:
            coro: The coroutine to schedule.
        """
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _require_client(self) -> TokenChainClient | None:
        return self._client

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #

    @filter.command("签到")
    async def cmd_checkin(self, event: AstrMessageEvent):
        """每日签到，随机获得游戏币。"""
        faucet = self._faucet_cfg()
        low = self._int_cfg(faucet, "checkin_min", 1)
        high = self._int_cfg(faucet, "checkin_max", 3)
        if low > high:
            low, high = high, low
        amount = random.randint(max(low, 0), max(high, 0))

        try:
            ok, balance = await self._store.try_checkin(
                self._user_key(event),
                event.get_sender_name(),
                self._day,
                amount,
            )
        except Exception as exc:
            logger.error(f"Check-in failed: {exc}")
            yield event.plain_result("签到失败，请稍后再试。")
            return

        if not ok:
            yield event.plain_result(
                f"今天已经签到过了，明天再来吧。\n当前余额：{balance} {self._symbol}",
            )
            return

        yield event.plain_result(
            f"签到成功，获得 {amount} {self._symbol}！\n"
            f"当前余额：{balance} {self._symbol}",
        )

    @filter.command("绑定钱包")
    async def cmd_bind(self, event: AstrMessageEvent, address: str = ""):
        """绑定用于收款的以太坊钱包地址。"""
        if not address:
            yield event.plain_result("用法：/绑定钱包 <0x 开头的钱包地址>")
            return
        if not TokenChainClient.is_valid_address(address):
            yield event.plain_result(
                "地址格式不正确，请检查是否为 0x 开头的 42 位地址。"
            )
            return

        checksummed = TokenChainClient.to_checksum(address)
        try:
            await self._store.set_wallet(
                self._user_key(event),
                checksummed,
                event.get_sender_name(),
            )
        except Exception as exc:
            logger.error(f"Wallet binding failed: {exc}")
            yield event.plain_result("绑定失败，请稍后再试。")
            return

        yield event.plain_result(
            f"钱包绑定成功：\n{checksummed}\n\n之后可直接使用 /提现 <数量> 提现到该地址。",
        )

    @filter.command("解绑钱包")
    async def cmd_unbind(self, event: AstrMessageEvent):
        """解除已绑定的钱包地址。"""
        try:
            await self._store.set_wallet(self._user_key(event), None)
        except Exception as exc:
            logger.error(f"Wallet unbinding failed: {exc}")
            yield event.plain_result("解绑失败，请稍后再试。")
            return
        yield event.plain_result("已解绑钱包地址。")

    @filter.command("钱包")
    async def cmd_wallet(self, event: AstrMessageEvent):
        """查看个人信息：余额、累计签到/提现与绑定地址。"""
        try:
            user = await self._store.get_user(self._user_key(event))
        except Exception as exc:
            logger.error(f"Wallet query failed: {exc}")
            yield event.plain_result("查询失败，请稍后再试。")
            return

        wallet = user.wallet or "未绑定（提现时需指定地址）"
        yield event.plain_result(
            f"余额：{user.balance} {self._symbol}\n"
            f"累计签到：{user.checkin_count} 次\n"
            f"累计提现：{user.total_withdrawn} {self._symbol}\n"
            f"绑定地址：{wallet}",
        )

    @filter.command("余额")
    async def cmd_balance(self, event: AstrMessageEvent):
        """查询余额；已绑定钱包时一并显示链上余额。"""
        symbol = self._symbol
        try:
            user = await self._store.get_user(self._user_key(event))
        except Exception as exc:
            logger.error(f"Balance query failed: {exc}")
            yield event.plain_result("查询失败，请稍后再试。")
            return

        if not user.wallet:
            yield event.plain_result(f"余额：{user.balance} {symbol}")
            return

        client = self._require_client()
        chain_line = f"链上余额：不可用（{self._chain_error}）"
        total_line = ""
        if client is not None:
            try:
                chain_balance = await client.token_balance(user.wallet)
                chain_line = f"链上余额：{chain_balance} {symbol}"
                total_line = f"\n合计：{user.balance + chain_balance} {symbol}"
            except ChainError as exc:
                logger.warning(f"Chain balance query failed: {exc}")
                chain_line = "链上余额：查询失败，请稍后再试"
        yield event.plain_result(
            f"插件余额：{user.balance} {symbol}\n{chain_line}{total_line}",
        )

    @filter.command("转账")
    async def cmd_transfer(self, event: AstrMessageEvent):
        """插件内转账：/转账 @某人 <数量>，仅转移插件余额，不上链。"""
        symbol = self._symbol
        self_id = str(event.get_self_id())
        at_segs = [
            seg
            for seg in event.get_messages()
            if isinstance(seg, At) and str(seg.qq) not in (self_id, "all")
        ]
        if not at_segs:
            yield event.plain_result("用法：/转账 @某人 <数量>（需要 @ 转账对象）")
            return
        if len({str(seg.qq) for seg in at_segs}) > 1:
            yield event.plain_result("一次只能转账给一个人。")
            return
        target_seg = at_segs[0]
        target_id = str(target_seg.qq)

        # On QQ the mention is spliced into message_str as " @nick(id) " and a
        # nickname may contain spaces, so positional command parameters are
        # unreliable here. The amount is the last purely numeric token instead.
        tokens = re.split(r"\s+", event.get_message_str().strip())
        amount_text = next((t for t in reversed(tokens) if t.isdigit()), "")
        if not amount_text or int(amount_text) < 1:
            yield event.plain_result(f"数量必须是正整数（{symbol} 不可拆分为小数）。")
            return
        amount = int(amount_text)

        sender_key = self._user_key(event)
        target_key = f"{event.get_platform_id()}:{target_id}"
        if target_key == sender_key:
            yield event.plain_result("不能转账给自己。")
            return

        try:
            ok, balance = await self._store.transfer_balance(
                sender_key,
                target_key,
                amount,
            )
        except Exception as exc:
            logger.error(f"Transfer failed: {exc}")
            yield event.plain_result("转账失败，请稍后再试。")
            return

        if not ok:
            yield event.plain_result(
                f"余额不足。当前余额 {balance} {symbol}，本次需要 {amount} {symbol}。",
            )
            return

        target_name = target_seg.name or target_id
        yield event.plain_result(
            f"已转账 {amount} {symbol} 给 @{target_name}。\n"
            f"你的余额：{balance} {symbol}",
        )

    @filter.command("提现")
    async def cmd_withdraw(
        self,
        event: AstrMessageEvent,
        arg1: str = "",
        arg2: str = "",
    ):
        """提现到链上钱包：/提现 <数量> 或 /提现 <地址> <数量>。"""
        symbol = self._symbol
        faucet = self._faucet_cfg()
        user_key = self._user_key(event)

        if not arg1:
            yield event.plain_result(
                "用法：\n"
                "  /提现 <数量>            提现到已绑定地址\n"
                "  /提现 <地址> <数量>     提现到指定地址",
            )
            return

        # `/提现 <地址> <数量>` vs `/提现 <数量>`: the first token decides.
        explicit_address: str | None = None
        if TokenChainClient.is_valid_address(arg1):
            explicit_address = TokenChainClient.to_checksum(arg1)
            amount_text = arg2
            if not amount_text:
                yield event.plain_result(f"请指定提现数量，例如：/提现 {arg1} 10")
                return
        elif arg1.startswith("0x"):
            yield event.plain_result(
                "地址格式不正确，请检查是否为 0x 开头的 42 位地址。"
            )
            return
        else:
            amount_text = arg1

        # Validate the amount before resolving the address, so a typo like
        # `/提现 abc` reports the real problem instead of "bind a wallet first".
        if not amount_text.isdigit():
            # decimals == 0 means the token is indivisible; reject fractional
            # input explicitly instead of silently truncating it.
            yield event.plain_result(f"数量必须是正整数（{symbol} 不可拆分为小数）。")
            return
        amount = int(amount_text)

        if explicit_address is not None:
            target = explicit_address
        else:
            try:
                user = await self._store.get_user(user_key)
            except Exception as exc:
                logger.error(f"Withdraw lookup failed: {exc}")
                yield event.plain_result("查询失败，请稍后再试。")
                return
            if not user.wallet:
                yield event.plain_result(
                    "你还没有绑定钱包。\n"
                    "请先 /绑定钱包 <地址>，或直接使用 /提现 <地址> <数量>。",
                )
                return
            target = user.wallet

        min_amount = self._int_cfg(faucet, "min_withdraw", 1)
        max_amount = self._int_cfg(faucet, "max_withdraw", 100)
        if amount < min_amount:
            yield event.plain_result(f"单次提现最少 {min_amount} {symbol}。")
            return
        if amount > max_amount:
            yield event.plain_result(f"单次提现最多 {max_amount} {symbol}。")
            return

        client = self._require_client()
        if client is None:
            logger.error(f"Withdraw rejected, chain unavailable: {self._chain_error}")
            yield event.plain_result("提现功能当前不可用，请联系管理员检查插件配置。")
            return

        limit = self._int_cfg(faucet, "daily_global_limit", 1000)
        day = self._day
        try:
            withdrawal_id, reason, value = await self._store.reserve_withdrawal(
                user_key,
                target,
                amount,
                day,
                limit,
            )
        except Exception as exc:
            logger.error(f"Withdraw reservation failed: {exc}")
            yield event.plain_result("提现失败，请稍后再试。")
            return

        if withdrawal_id is None:
            if reason == "insufficient_balance":
                yield event.plain_result(
                    f"余额不足。当前余额 {value} {symbol}，本次需要 {amount} {symbol}。",
                )
            else:
                yield event.plain_result(
                    f"今日全局提现额度不足。\n"
                    f"剩余额度：{value} {symbol}（每日上限 {limit} {symbol}），"
                    f"请明天再试。",
                )
            return

        # The balance is already debited. From here every failure path must
        # either broadcast the transfer or refund the reservation.
        try:
            result = await client.transfer(target, amount, wait_for_receipt=False)
        except ChainError as exc:
            await self._store.refund(withdrawal_id, str(exc))
            logger.error(f"Withdraw broadcast failed: {exc}")
            yield event.plain_result(
                f"提现失败，{amount} {symbol} 已退回余额。\n原因：{exc}",
            )
            return
        except Exception as exc:
            await self._store.refund(withdrawal_id, str(exc))
            logger.error(f"Withdraw broadcast crashed: {exc}")
            yield event.plain_result(f"提现失败，{amount} {symbol} 已退回余额。")
            return

        await self._store.mark_sent(withdrawal_id, result.tx_hash)
        self._spawn(
            self._confirm_withdrawal(
                withdrawal_id,
                result.tx_hash,
                amount,
                event.unified_msg_origin,
            ),
        )

        yield event.plain_result(
            f"提现已提交：{amount} {symbol} → {target}\n"
            f"交易哈希：{result.tx_hash}\n"
            f"{self._explorer_tx(result.tx_hash)}\n\n"
            f"链上确认后会再通知你。",
        )

    @filter.command("提现记录")
    async def cmd_history(self, event: AstrMessageEvent):
        """查看最近的提现记录。"""
        try:
            rows = await self._store.list_withdrawals(self._user_key(event), limit=5)
        except Exception as exc:
            logger.error(f"History query failed: {exc}")
            yield event.plain_result("查询失败，请稍后再试。")
            return

        if not rows:
            yield event.plain_result("暂无提现记录。")
            return

        status_text = {
            "pending": "处理中",
            "sent": "已广播",
            "confirmed": "已确认",
            "failed": "失败（已退款）",
        }
        lines = ["最近 5 条提现记录："]
        for row in rows:
            tx = row["tx_hash"] or ""
            short_tx = f" {tx[:10]}…" if tx else ""
            lines.append(
                f"· {row['day']} {row['amount']} {self._symbol} "
                f"[{status_text.get(row['status'], row['status'])}]{short_tx}",
            )
        yield event.plain_result("\n".join(lines))

    @filter.command("游戏币")
    async def cmd_token_info(self, event: AstrMessageEvent):
        """查看代币信息、分发钱包余额与今日剩余额度。"""
        client = self._require_client()
        if client is None:
            yield event.plain_result(f"链上功能当前不可用：{self._chain_error}")
            return

        limit = self._int_cfg(self._faucet_cfg(), "daily_global_limit", 1000)
        try:
            used = await self._store.daily_used(self._day)
        except Exception as exc:
            logger.error(f"Quota query failed: {exc}")
            yield event.plain_result("查询失败，请稍后再试。")
            return

        try:
            info = await client.token_info()
            pool_balance = await client.token_balance()
        except ChainError as exc:
            logger.error(f"Token info query failed: {exc}")
            yield event.plain_result("链上查询失败，请稍后再试。")
            return

        yield event.plain_result(
            f"{info['name']}（{info['symbol']}）\n"
            f"总量：{info['total_supply']}\n"
            f"合约地址：{client.token_address}\n"
            f"分发钱包余额：{pool_balance}\n"
            f"今日剩余提现额度：{max(limit - used, 0)}/{limit}",
        )

    @filter.command("排行榜")
    async def cmd_rank(self, event: AstrMessageEvent):
        """链上持有排行榜，仅统计已绑定钱包的用户。"""
        client = self._require_client()
        if client is None:
            yield event.plain_result(f"链上功能当前不可用：{self._chain_error}")
            return

        try:
            bound = await self._store.list_bound_wallets()
        except Exception as exc:
            logger.error(f"Rank lookup failed: {exc}")
            yield event.plain_result("查询失败，请稍后再试。")
            return
        if not bound:
            yield event.plain_result("还没有用户绑定钱包，暂无排行榜。")
            return

        try:
            supply = int((await client.token_info())["total_supply"])
        except ChainError as exc:
            logger.error(f"Rank token info failed: {exc}")
            yield event.plain_result("链上查询失败，请稍后再试。")
            return

        # Balances come from the bound addresses, not from a holders API:
        # standard RPC nodes cannot enumerate token holders, and the board
        # only shows bound users anyway. Capped concurrency keeps a long
        # binding list from tripping public-RPC rate limits.
        sem = asyncio.Semaphore(8)

        async def query_balance(wallet: str) -> int | None:
            async with sem:
                try:
                    return await client.token_balance(wallet)
                except ChainError as exc:
                    logger.warning(f"Rank balance failed for {wallet}: {exc}")
                    return None

        balances = await asyncio.gather(*[query_balance(w) for _, _, w in bound])

        entries = []
        failed = 0
        for (user_key, name, wallet), balance in zip(bound, balances):
            if balance is None:
                failed += 1
            elif balance > 0:
                display = name or user_key.split(":", 1)[-1]
                entries.append((display, wallet, balance))
        entries.sort(key=lambda e: e[2], reverse=True)

        if not entries:
            if failed:
                yield event.plain_result("链上查询失败，请稍后再试。")
            else:
                yield event.plain_result(
                    "已绑定钱包的用户当前链上均无持仓，暂无排行榜。",
                )
            return

        lines = [f"{self._symbol} 持有排行榜（仅统计已绑定钱包的用户）"]
        for i, (display, wallet, balance) in enumerate(entries[:10], start=1):
            pct = balance / supply * 100 if supply else 0.0
            lines.append(
                f"{i}. {display} {wallet[:6]}…{wallet[-4:]}："
                f"{balance} {self._symbol}（{pct:.2f}%）",
            )
        if failed:
            lines.append(f"另有 {failed} 个地址查询失败，未计入。")
        yield event.plain_result("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Background confirmation
    # ------------------------------------------------------------------ #

    async def _confirm_withdrawal(
        self,
        withdrawal_id: int,
        tx_hash: str,
        amount: int,
        umo: str,
    ) -> None:
        """Wait for a broadcast withdrawal to mine and report the outcome.

        A revert refunds the user. A timeout does not: the transaction is still
        in the mempool and may land later, so the row stays ``sent`` and keeps
        holding its share of the daily quota.

        Args:
            withdrawal_id: Row id of the reserved withdrawal.
            tx_hash: Hash of the broadcast transaction.
            amount: Token count, used in the notification text.
            umo: Unified message origin used to push the notification.
        """
        client = self._require_client()
        if client is None:
            return

        timeout = float(self._int_cfg(self._chain_cfg(), "receipt_timeout", 180))
        try:
            succeeded, _ = await client.wait_for_receipt(tx_hash, timeout=timeout)
        except asyncio.CancelledError:
            raise
        except ChainError as exc:
            # Timed out waiting: the transaction may still mine, so the row
            # stays `sent` and is neither confirmed nor refunded here.
            logger.warning(f"Receipt wait failed for {tx_hash}: {exc}")
            return

        if succeeded:
            await self._store.mark_confirmed(withdrawal_id)
            text = (
                f"提现已确认：{amount} {self._symbol} 到账。\n"
                f"{self._explorer_tx(tx_hash)}"
            )
        else:
            await self._store.refund(withdrawal_id, f"reverted: {tx_hash}")
            text = (
                f"提现失败：交易被链上回滚，{amount} {self._symbol} 已退回余额。\n"
                f"{self._explorer_tx(tx_hash)}"
            )

        try:
            await self.context.send_message(umo, MessageChain().message(text))
        except Exception as exc:
            logger.warning(f"Failed to push withdrawal result: {exc}")
